import os
import json
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration

DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
MAX_RESEARCH_LOOPS = 3

# Models served via OpenAI-compatible cloud providers
CLOUD_MODELS = {
        "dola-seed-2-1-turbo-260628": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "api_key_env": "ARK_API_KEY",
    },
    "seed-2-0-lite-260228": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "api_key_env": "ARK_API_KEY",
    },
    "glm-4-7-251222": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "api_key_env": "ARK_API_KEY",
    },
    "moonshot-v1-8k": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "qwen-max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}
# Backwards-compatible aliases
ARK_MODELS = set(CLOUD_MODELS.keys())
def _llm(config):
    cfg = (config or {}).get("configurable", {}) or {}
    model = cfg.get("model_name") or DEFAULT_MODEL
    temperature = cfg.get("temperature", 0)
    if model in CLOUD_MODELS:
        provider = CLOUD_MODELS[model]
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=provider["base_url"],
            api_key=os.environ.get(provider["api_key_env"]),
        )
    return ChatOllama(model=model, temperature=temperature)
    return ChatOllama(model=model, temperature=temperature)


def _search_tool(config):
    cfg = (config or {}).get("configurable", {}) or {}
    max_results = cfg.get("max_search_results", 5)
    return TavilySearch(max_results=max_results, search_depth="advanced")


# ---------------------------------------------------------------------------
# Node: generate_queries
# ---------------------------------------------------------------------------
def generate_queries(state: OverallState, config) -> dict:
    write = get_stream_writer()
    question = state.get("input", "")
    knowledge_gap = state.get("knowledge_gap", "")
    loop_count = state.get("research_loop_count", 0)

    write({"type": "status", "node": "generate_queries",
           "message": f"Generating search queries (round {loop_count + 1})…"})

    if loop_count == 0:
        user_msg = (
            f"Generate 3 focused search queries to research the following question.\n\n"
            f"Question: {question}\n\n"
            f"Respond ONLY with a JSON array of strings, e.g. [\"query 1\", \"query 2\", \"query 3\"]. "
            f"No explanation, no markdown, just the JSON array."
        )
    else:
        user_msg = (
            f"We are researching this question: {question}\n\n"
            f"We have already gathered some information but there is a knowledge gap:\n"
            f"{knowledge_gap}\n\n"
            f"Generate 2 focused follow-up search queries to fill this gap.\n"
            f"Respond ONLY with a JSON array of strings. No explanation, no markdown."
        )

    system = SystemMessage(content=(
        "You are a research assistant. Your job is to generate precise, targeted web search queries. "
        "Always respond with valid JSON only."
    ))
    response = _llm(config).invoke([system, HumanMessage(content=user_msg)])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        queries = json.loads(raw.strip())
        if not isinstance(queries, list):
            queries = [question]
    except Exception:
        queries = [question]

    write({"type": "queries", "queries": queries, "round": loop_count + 1})

    return {
        "query_list": queries,
        "search_query": queries,
        "research_loop_count": loop_count + 1,
    }


# ---------------------------------------------------------------------------
# Node: web_search  (queries run in parallel)
# ---------------------------------------------------------------------------
async def web_search(state: OverallState, config) -> dict:
    write = get_stream_writer()
    queries = state.get("query_list", []) or [state.get("input", "")]
    tool = _search_tool(config)

    write({"type": "status", "node": "web_search",
           "message": f"Searching {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} in parallel…"})

    async def _search_one(query: str):
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: tool.invoke({"query": query})
            )
            results = raw.get("results", []) if isinstance(raw, dict) else []
            docs, sources = [], []
            for r in results:
                content = r.get("content", "").strip()
                url = r.get("url", "").strip()
                if content:
                    docs.append(f"[Source: {url}]\n{content}")
                if url:
                    sources.append(url)
            return docs, sources
        except Exception as e:
            return [f"[Search failed for query '{query}': {e}]"], []

    async def _run_all():
        return await asyncio.gather(*[_search_one(q) for q in queries])

    try:
        results = await _run_all()
    except Exception as e:
        results = [([f"[Search error: {e}]"], [])]


    all_docs: list[str] = []
    all_sources: list[str] = []
    for docs, sources in results:
        all_docs.extend(docs)
        all_sources.extend(sources)

    # Deduplicate sources preserving order
    seen: set[str] = set()
    deduped_sources = []
    for s in all_sources:
        if s not in seen:
            seen.add(s)
            deduped_sources.append(s)

    write({"type": "sources", "sources": deduped_sources})

    return {
        "documents": all_docs,
        "sources_gathered": deduped_sources,
        "number_of_ran_queries": state.get("number_of_ran_queries", 0) + len(queries),
        "web_research_result": all_docs,
    }


# ---------------------------------------------------------------------------
# Node: reflect
# ---------------------------------------------------------------------------
def reflect(state: OverallState, config) -> dict:
    write = get_stream_writer()
    question = state.get("input", "")
    documents = state.get("documents", [])
    loop_count = state.get("research_loop_count", 0)
    max_loops = (config or {}).get("configurable", {}).get("max_research_loops", MAX_RESEARCH_LOOPS)

    write({"type": "status", "node": "reflect",
           "message": f"Evaluating research quality (round {loop_count})…"})

    if loop_count >= max_loops:
        write({"type": "status", "node": "reflect",
               "message": "Max research rounds reached. Proceeding to answer."})
        return {"is_sufficient": True, "knowledge_gap": ""}

    docs_text = "\n\n---\n\n".join(documents) if documents else "No documents gathered."

    user_msg = (
        f"You are evaluating whether we have enough information to answer this question:\n"
        f"{question}\n\n"
        f"Here is the research gathered so far:\n{docs_text}\n\n"
        f"Respond ONLY with a JSON object with these fields:\n"
        f"  is_sufficient: boolean\n"
        f"  knowledge_gap: string — what is missing; empty string if sufficient\n\n"
        f"Example: {{\"is_sufficient\": false, \"knowledge_gap\": \"Missing recent data on X\"}}\n"
        f"No explanation, no markdown, just the JSON object."
    )

    system = SystemMessage(content=(
        "You are a critical research evaluator. Be strict. Respond with valid JSON only."
    ))
    response = _llm(config).invoke([system, HumanMessage(content=user_msg)])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        is_sufficient = bool(result.get("is_sufficient", False))
        knowledge_gap = result.get("knowledge_gap", "")
    except Exception:
        is_sufficient = True
        knowledge_gap = ""

    if is_sufficient:
        write({"type": "status", "node": "reflect", "message": "Research sufficient. Writing answer…"})
    else:
        write({"type": "gap", "knowledge_gap": knowledge_gap})

    return {"is_sufficient": is_sufficient, "knowledge_gap": knowledge_gap}


# ---------------------------------------------------------------------------
# Node: answer
# ---------------------------------------------------------------------------
def answer(state: OverallState, config) -> dict:
    write = get_stream_writer()
    question = state.get("input", "")
    documents = state.get("documents", [])
    loops = state.get("research_loop_count", 1)

    write({"type": "status", "node": "answer",
           "message": f"Writing answer from {len(documents)} documents ({loops} round(s) of research)…"})

    docs_text = "\n\n---\n\n".join(documents) if documents else "No research documents available."

    system = SystemMessage(content=(
        "You are a knowledgeable, precise research assistant. "
        "Answer the user's question using ONLY the provided research documents. "
        "Be thorough but concise. If the documents don't fully answer the question, "
        "say so clearly and share what you do know. "
        "Do not fabricate information."
    ))
    user_msg = (
        f"Question: {question}\n\n"
        f"Research documents ({loops} search round(s)):\n\n{docs_text}\n\n"
        f"Provide a complete, well-structured answer based on the documents above."
    )

    response = _llm(config).invoke([system, HumanMessage(content=user_msg)])
    return {"response": response.content}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_reflect(state: OverallState) -> str:
    return "answer" if state.get("is_sufficient", False) else "generate_queries"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
builder = StateGraph(OverallState, context_schema=Configuration)
builder.add_node("generate_queries", generate_queries)
builder.add_node("web_search", web_search)
builder.add_node("reflect", reflect)
builder.add_node("answer", answer)

builder.set_entry_point("generate_queries")
builder.add_edge("generate_queries", "web_search")
builder.add_edge("web_search", "reflect")
builder.add_conditional_edges("reflect", route_after_reflect, {
    "answer": "answer",
    "generate_queries": "generate_queries",
})
builder.add_edge("answer", END)

graph = builder.compile()