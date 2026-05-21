# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangGraph-based AI agent ("InotGrumpy"). Python 3.11+. The compiled graph is exposed via three surfaces:

1. **LangGraph CLI / Studio** — configured by `langgraph.json` (entry: `graphs.agent`).
2. **FastAPI + LangServe** — `server.py` mounts the same `graph` object at `/agent` and exposes a JSON API.
3. **Custom UI** — `static/index.html` served at `/`, talking to `/agent/invoke`. Same backend as the playground; just a nicer shell.

All surfaces import the same compiled `graph` from the `agent` package, so changes to `agent/graph.py` propagate to all of them.

## Commands

```bash
# Install (editable install picks up pyproject.toml deps)
pip install -e .
pip install -r requirements.txt   # adds langgraph-cli[inmem] for Studio

# Run LangGraph Studio / dev server (reads langgraph.json)
langgraph dev

# Run the FastAPI/LangServe server + custom UI
uvicorn server:app --reload

# Lint
ruff check .

# Tests
pytest
pytest path/to/test_file.py::test_name   # single test
```

Environment variables are loaded from `.env`. `langgraph.json` references it for `langgraph dev`; `server.py` calls `load_dotenv()` at import time so `uvicorn` picks it up too.

## Architecture

### Graph (`agent/graph.py`)
Builds a `StateGraph` with two nodes — `search` and `answer` — wired via `add_conditional_edges("search", decide_route)`. Entry point is `search`; `answer` terminates at `END`. The compiled `graph` is the public artifact other modules import.

`decide_route` always returns `"answer"`, so every query goes through Tavily search first and then the LLM. Keep that contract — the UI copy ("Every question is answered with a Tavily web search first.") depends on it.

The LLM is **not** instantiated at module top level. `_llm(config)` builds a fresh `ChatOllama` per-call using `config["configurable"]["model_name"]`, falling back to `DEFAULT_MODEL`. This is what makes the UI's model dropdown work — selections come in via LangServe's `configurable` config.

`DEFAULT_MODEL` is the public default; `server.py` imports it for the `/models` endpoint response.

The builder uses the typed form: `StateGraph(OverallState, context_schema=Configuration)`. New node functions should return dicts whose keys match `OverallState`. Node signatures that need runtime config must accept `(state, config)` — LangGraph passes config as the second positional arg.

**Imports in graph entry files must be absolute** (`from agent.state import ...`, not `from .state import ...`). LangGraph CLI loads `./agent/graph.py` as a standalone script with no parent package, so relative imports fail at load time with `attempted relative import with no known parent package`.

### State (`agent/state.py`)
Defines TypedDicts shaped for a **research-loop agent** (query generation → web search → reflection → follow-up). `OverallState.follow_up_queries` uses `Annotated[list[str], operator.add]` so LangGraph reducers append rather than overwrite — preserve this when adding accumulator fields.

`Configuration` lives in `agent/configuration.py` (single source of truth — do not redefine it elsewhere). It exposes `model_name`, `temperature`, etc. — anything addable to `config.configurable` from the client.

When adding new graph entries to `langgraph.json`, use paths relative to the repo root in the form `./agent/<file>.py:<exported_var>`.

### Server (`server.py`)
- `add_routes(app, graph, path="/agent", config_keys=["configurable"])` — the `config_keys` arg is required so clients can pass `model_name` (or any other `Configuration` field) per request. Without it LangServe drops the field.
- `/` → serves `static/index.html` (the custom UI).
- `/static/*` → mounted `StaticFiles`.
- `/models` → returns the SEA-LION catalog annotated with which Ollama tags are installed locally (queries `$OLLAMA_HOST/api/tags`, default `http://localhost:11434`).
- `/avatar.jpeg` → serves `agent/avatar.jpeg` (InotGrumpy portrait used in the UI header).

### UI (`static/index.html`)
Single-file vanilla HTML + Tailwind CDN + Inter/Fraunces fonts. Visual language mirrors <https://sagelion.vercel.app/> — cream background, charcoal ink, no gradients, Unicode ornaments (◌ ◇ ✦ 🌒) as accents.

Calls `/models` on load to build the dropdown, then POSTs to `/agent/invoke` with `{ input: { input }, config: { configurable: { model_name } } }`. No build step.
