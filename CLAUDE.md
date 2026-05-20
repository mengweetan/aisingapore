# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangGraph-based AI agent. Python 3.11+. The compiled graph is exposed via two surfaces:

1. **LangGraph CLI / Studio** — configured by `langgraph.json` (entry: `graphs.agent`).
2. **FastAPI + LangServe** — `server.py` mounts the same `graph` object at `/agent`.

Both surfaces import the same compiled `graph` from the `agent` package, so changes to `agent/graph.py` propagate to both.

## Commands

```bash
# Install (editable install picks up pyproject.toml deps)
pip install -e .
pip install -r requirements.txt   # adds langgraph-cli[inmem] for Studio

# Run LangGraph Studio / dev server (reads langgraph.json)
langgraph dev

# Run the FastAPI/LangServe server
uvicorn server:app --reload

# Lint
ruff check .

# Tests
pytest
pytest path/to/test_file.py::test_name   # single test
```

Environment variables are loaded from `.env` (referenced by `langgraph.json`).

## Architecture

### Graph (`agent/graph.py`)
Builds a `StateGraph` with two nodes — `search` and `answer` — wired via `add_conditional_edges("search", decide_route)`. Entry point is `search`; `answer` terminates at `END`. The compiled `graph` is the public artifact other modules import.

The builder uses the typed form: `StateGraph(OverallState, context_schema=Configuration)`. New node functions should return dicts whose keys match `OverallState` so the typed schema stays accurate.

**Imports in graph entry files must be absolute** (`from agent.state import ...`, not `from .state import ...`). LangGraph CLI loads `./agent/graph.py` as a standalone script with no parent package, so relative imports fail at load time with `attempted relative import with no known parent package`.

### State (`agent/state.py`)
Defines TypedDicts shaped for a **research-loop agent** (query generation → web search → reflection → follow-up). `OverallState.follow_up_queries` uses `Annotated[list[str], operator.add]` so LangGraph reducers append rather than overwrite — preserve this when adding accumulator fields.

`Configuration` lives in `agent/configuration.py` (single source of truth — do not redefine it elsewhere).

When adding new graph entries to `langgraph.json`, use paths relative to the repo root in the form `./agent/<file>.py:<exported_var>`.
