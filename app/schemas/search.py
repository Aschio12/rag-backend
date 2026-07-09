from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    doc_id: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
