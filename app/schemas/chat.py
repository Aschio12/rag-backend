from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class Source(BaseModel):
    title: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    conversation_id: str
