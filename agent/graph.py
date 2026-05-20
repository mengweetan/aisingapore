from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration

llm = ChatOllama(
    model="aisingapore/Gemma-SEA-LION-v4-4B-VL:latest",
    temperature=0,
)

search_tool = TavilySearch(max_results=5, search_depth="advanced")

def search_node(state):
    query = state.get("input", "")
    raw = search_tool.invoke({"query": query})
    results = raw.get("results", []) if isinstance(raw, dict) else []
    documents = [r.get("content", "") for r in results]
    sources = [r.get("url", "") for r in results]
    return {"documents": documents, "sources_gathered": sources}

def answer_node(state):
    question = state.get("input", "")
    docs = state.get("documents", [])
    prompt = f"Use these docs to answer the question.\n\nDocs: {docs}\n\nQuestion: {question}"
    reply = llm.invoke([HumanMessage(content=prompt)])
    return {"response": reply.content}

#decide route function
def decide_route(state):
    # your routing logic; must return a node name (e.g., "search", "answer")
    if "search" in state.get("input", "").lower():
        return "search"
    return "answer"

builder = StateGraph(OverallState, context_schema=Configuration)
builder.add_node("search", search_node)
builder.add_node("answer", answer_node)
builder.add_conditional_edges("search", decide_route)
builder.add_edge("answer", END)
builder.set_entry_point("search")
graph = builder.compile()

from langchain_core.runnables import Runnable
print(type(graph))
print(isinstance(graph, Runnable))
