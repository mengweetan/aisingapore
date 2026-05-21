from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from agent.graph import graph, DEFAULT_MODEL  # the compiled LangGraph Runnable

app = FastAPI()
add_routes(app, graph, path="/agent", config_keys=["configurable"])

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Catalog from https://sea-lion.ai/models/ — display labels for known Ollama tags.
SEA_LION_CATALOG = [
    ("aisingapore/Apertus-SEA-LION-v4-8B-IT",   "Apertus-SEA-LION v4 · 8B · IT"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-IT",    "Gemma-SEA-LION v4 · 27B · IT"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-VL",    "Gemma-SEA-LION v4 · 27B · VL"),
    ("aisingapore/Gemma-SEA-LION-v4-4B-VL",     "Gemma-SEA-LION v4 · 4B · VL"),
    ("aisingapore/Qwen-SEA-LION-v4-32B-IT",     "Qwen-SEA-LION v4 · 32B · IT"),
    ("aisingapore/Qwen-SEA-LION-v4-8B-VL",      "Qwen-SEA-LION v4 · 8B · VL"),
    ("aisingapore/Qwen-SEA-LION-v4-4B-VL",      "Qwen-SEA-LION v4 · 4B · VL"),
    ("aisingapore/Llama-SEA-LION-v3.5-70B-R",   "Llama-SEA-LION v3.5 · 70B · Reasoning"),
    ("aisingapore/Llama-SEA-LION-v3.5-8B-R",    "Llama-SEA-LION v3.5 · 8B · Reasoning"),
]

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/avatar.jpeg")
def avatar():
    return FileResponse(Path(__file__).parent / "agent" / "avatar.jpeg", media_type="image/jpeg")

@app.get("/models")
async def list_models():
    """Return the SEA-LION catalog annotated with which models are installed locally."""
    installed: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            for m in r.json().get("models", []):
                name = m.get("name", "")
                installed.add(name)
                installed.add(name.split(":", 1)[0])  # strip :tag suffix
    except Exception:
        pass

    models = [
        {"id": tag, "label": label, "installed": tag in installed}
        for tag, label in SEA_LION_CATALOG
    ]
    return {"default": DEFAULT_MODEL, "models": models}
