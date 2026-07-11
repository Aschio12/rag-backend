from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    top_k: int = 5
    hybrid: bool = False
    rerank: bool = False


class Source(BaseModel):
    doc_id: str = ""
    text: str = ""
    score: float = 0.0
    page_number: int = 0
    bbox: str = ""
    filename: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    conversation_id: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool = False
    collection_id: str | None = None


class SearchResult(BaseModel):
    id: str = ""
    text: str = ""
    score: float = 0.0
    doc_id: str = ""
    page_number: int = 0
    filename: str = ""
    bbox: str = ""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = []
    total: int = 0
