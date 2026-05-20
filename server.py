from fastapi import FastAPI
from langserve import add_routes
from agent.graph import graph  # the compiled LangGraph Runnable

app = FastAPI()
add_routes(app, graph, path="/agent")
