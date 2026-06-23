from dotenv import load_dotenv
load_dotenv()

import os
import json
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from agent.graph import graph, DEFAULT_MODEL

app = FastAPI()
add_routes(app, graph, path="/agent", config_keys=["configurable"])

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SEA_LION_CATALOG = [
    ("aisingapore/Apertus-SEA-LION-v4-8B-IT",  "Apertus-SEA-LION v4 · 8B · IT"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-IT",   "Gemma-SEA-LION v4 · 27B · IT"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-VL",   "Gemma-SEA-LION v4 · 27B · VL"),
    ("aisingapore/Gemma-SEA-LION-v4-4B-VL",    "Gemma-SEA-LION v4 · 4B · VL"),
    ("aisingapore/Qwen-SEA-LION-v4-32B-IT",    "Qwen-SEA-LION v4 · 32B · IT"),
    ("aisingapore/Qwen-SEA-LION-v4-8B-VL",     "Qwen-SEA-LION v4 · 8B · VL"),
    ("aisingapore/Qwen-SEA-LION-v4-4B-VL",     "Qwen-SEA-LION v4 · 4B · VL"),
    ("aisingapore/Llama-SEA-LION-v3.5-70B-R",  "Llama-SEA-LION v3.5 · 70B · Reasoning"),
    ("aisingapore/Llama-SEA-LION-v3.5-8B-R",   "Llama-SEA-LION v3.5 · 8B · Reasoning"),
    ("glm-4-7-251222", "GLM-4.7 · BytePlus Ark ☁"),
]

# Cloud models are always "installed" (no local pull needed)
CLOUD_MODELS = {"glm-4-7-251222"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/avatar.jpeg")
def avatar():
    return FileResponse(Path(__file__).parent / "agent" / "avatar.jpeg", media_type="image/jpeg")


@app.get("/models")
async def list_models():
    installed: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            for m in r.json().get("models", []):
                name = m.get("name", "")
                installed.add(name)
                installed.add(name.split(":", 1)[0])
    except Exception:
        pass

    models = [
        {
            "id": tag,
            "label": label,
            "installed": tag in installed or tag in CLOUD_MODELS,
        }
        for tag, label in SEA_LION_CATALOG
    ]
    return {"default": DEFAULT_MODEL, "models": models}


@app.post("/stream")
async def agent_stream(request: Request):
    """SSE endpoint — streams node-level progress + final answer."""
    body = await request.json()
    user_input = body.get("input", {}).get("input", "")
    config = body.get("config", {})

    async def event_generator():
        try:
            async for chunk in graph.astream(
                {"input": user_input},
                config=config,
                stream_mode=["updates", "custom"],
            ):
                # custom events from get_stream_writer()
                if isinstance(chunk, tuple) and chunk[0] == "custom":
                    data = chunk[1]
                    yield f"data: {json.dumps(data)}\n\n"

                # node completion updates
                elif isinstance(chunk, tuple) and chunk[0] == "updates":
                    node_updates = chunk[1]
                    for node_name, state_delta in node_updates.items():
                        # Emit queries when generate_queries completes
                        if node_name == "generate_queries" and "query_list" in state_delta:
                            yield f"data: {json.dumps({'type': 'queries_done', 'queries': state_delta['query_list']})}\n\n"
                        # Emit sources when web_search completes
                        if node_name == "web_search" and "sources_gathered" in state_delta:
                            yield f"data: {json.dumps({'type': 'sources_done', 'sources': state_delta['sources_gathered']})}\n\n"
                        # Emit final answer when answer node completes
                        if node_name == "answer" and "response" in state_delta:
                            yield f"data: {json.dumps({'type': 'answer', 'response': state_delta['response'], 'sources': state_delta.get('sources_gathered', [])})}\n\n"

                # dict format (LangGraph sometimes skips the tuple wrapper)
                elif isinstance(chunk, dict):
                    for node_name, state_delta in chunk.items():
                        if isinstance(state_delta, dict):
                            if node_name == "answer" and "response" in state_delta:
                                yield f"data: {json.dumps({'type': 'answer', 'response': state_delta['response']})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
