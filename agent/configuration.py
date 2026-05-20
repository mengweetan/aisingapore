from typing_extensions import TypedDict

class Configuration(TypedDict, total=False):
    model_name: str
    max_search_results: int
    temperature: float
    search_api_key: str
