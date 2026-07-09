from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int = 0


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentResponse
