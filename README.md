# myclaw

LangGraph agent that answers questions by combining **Tavily** web search with the **`aisingapore/Gemma-SEA-LION-v4-4B-VL`** model running locally via Ollama. Tuned for Southeast Asian languages; runs fully offline aside from the search call.

## Setup

```bash
# 1. Python deps
pip install -e .

# 2. Pull the model (one-time)
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest

# 3. Env vars
echo 'TAVILY_API_KEY=tvly-...' > .env   # get a free key at https://app.tavily.com
```

## Run

```bash
langgraph dev                      # LangGraph Studio (recommended)
uvicorn server:app --reload        # FastAPI + LangServe at /agent
```

## Troubleshooting

- **`attempted relative import with no known parent package`** when running `langgraph dev` — the CLI loads `./agent/graph.py` as a script. Use absolute imports (`from agent.state import ...`), not relative (`from .state import ...`).

See [`CLAUDE.md`](./CLAUDE.md) for architecture notes.
