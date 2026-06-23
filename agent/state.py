# agent/state.py
from typing import Annotated
from typing_extensions import TypedDict
import operator

class OverallState(TypedDict, total=False):
    input: str
    response: str
    documents: Annotated[list[str], operator.add]
    messages: list
    query_list: list[str]
    search_query: list[str]
    web_research_result: list[str]
    sources_gathered: list[str]
    research_loop_count: int
    number_of_ran_queries: int
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list[str], operator.add]

class QueryGenerationState(TypedDict):
    query_list: list[str]

class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list[str], operator.add]
    research_loop_count: int
    number_of_ran_queries: int

class WebSearchState(TypedDict):
    search_query: str
    id: str
