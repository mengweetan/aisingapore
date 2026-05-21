# InotGrumpy

A patient little LangGraph agent that answers questions by combining **Tavily** web search with **SEA-LION** models running locally via **Ollama**. Tuned for Southeast Asian languages; runs fully offline aside from the search call.

## Setup

```bash
# 1. Python deps
pip install -e .

# 2. Pull at least one SEA-LION model (one-time)
ollama pull aisingapore/Qwen-SEA-LION-v4-32B-IT
# or a lighter one:
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL

# 3. Env vars
echo 'TAVILY_API_KEY=tvly-...' > .env   # get a free key at https://app.tavily.com
```

## Run

```bash
langgraph dev                      # LangGraph Studio
uvicorn server:app --reload        # FastAPI + LangServe + custom UI
```

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

See [`CLAUDE.md`](./CLAUDE.md) for architecture notes.
