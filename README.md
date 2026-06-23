# InotGrumpy

A patient little LangGraph agent that answers questions by combining **Tavily** web search with **SEA-LION** models running locally via **Ollama**. Tuned for Southeast Asian languages; runs fully offline aside from the search call.

## Prerequisites

- **Python 3.11+** — required by `langgraph-cli[inmem]`. Check with `python3 --version`. If you're on 3.9/3.10, download 3.11 from [python.org/downloads](https://www.python.org/downloads/) before continuing.
- **Ollama** — running locally at `http://localhost:11434`. Install from [ollama.com](https://ollama.com).
- **Tavily API key** — free at [app.tavily.com](https://app.tavily.com).

## Setup

```bash
# 1. Create a virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Python deps (always use the venv's pip, not the system one)
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt   # adds langgraph-cli[inmem]

# 2. Pull at least one SEA-LION model (one-time)
ollama pull aisingapore/Qwen-SEA-LION-v4-32B-IT
# or a lighter one:
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL

# 3. Env vars
echo 'TAVILY_API_KEY=tvly-...' > .env   # get a free key at https://app.tavily.com
```

## Run

Always activate the venv first (`source .venv/bin/activate`), or call the binaries directly:

```bash
.venv/bin/langgraph dev                      # LangGraph Studio
.venv/bin/uvicorn server:app --reload        # FastAPI + LangServe + custom UI
```

> **Why the explicit path?** macOS may have an older `langgraph` or `uvicorn` on the system `PATH`. Using `.venv/bin/` ensures you're running the right version.

Then open:

- **http://127.0.0.1:8000/** — InotGrumpy custom UI (model dropdown + chat)
- **http://127.0.0.1:8000/agent/playground** — LangServe playground
- **http://127.0.0.1:8000/agent/invoke** — JSON API used by both

## Picking a model from the UI

The header dropdown is populated from `/models`, which:

1. Hits Ollama's `/api/tags` to see which SEA-LION variants are pulled locally.
2. Shows installed ones as selectable; the rest of the catalog from <https://sea-lion.ai/models/> is listed as "Not pulled".

The chosen model is passed per-request via LangServe's `config.configurable.model_name` — see `agent/graph.py:_llm()`.

## Troubleshooting

- **`attempted relative import with no known parent package`** when running `langgraph dev` — the CLI loads `./agent/graph.py` as a script. Use absolute imports (`from agent.state import ...`), not relative (`from .state import ...`).
- **`Did not find tavily_api_key`** when running uvicorn — `server.py` calls `load_dotenv()` at startup, so `TAVILY_API_KEY` must be in `.env` at the repo root.
- **`The in-mem server requires Python 3.11 or higher ... you are currently using Python 3.9`** even after installing 3.11 — your venv wasn't created with 3.11, or you're calling the system `langgraph` binary instead of the venv one. Fix: `rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Then use `.venv/bin/langgraph dev`.
- **`which langgraph` returns `/usr/local/bin/langgraph`** — that's the system install, not your venv. Always use `.venv/bin/langgraph dev` or activate the venv first.
- **`ModuleNotFoundError: No module named 'fastapi'`** when running uvicorn — run `.venv/bin/pip install fastapi uvicorn langserve httpx` and use `.venv/bin/uvicorn server:app --reload`.

See [`CLAUDE.md`](./CLAUDE.md) for architecture notes.
