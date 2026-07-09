import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.ingestion import ingest_document
from app.rag import answer, answer_stream
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.store import list_documents
from app.vector_store import search as vector_search

router = APIRouter(prefix="/api/v1")


@router.get("/")
async def api_root():
    return {"message": "RAG API v1"}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile):
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = ingest_document(str(file_path))

    return DocumentUploadResponse(
        message="Document uploaded and indexed",
        document=DocumentResponse(
            id=doc["id"],
            filename=doc["filename"],
            status=doc["status"],
            chunk_count=doc["chunk_count"],
        ),
    )


@router.get("/documents")
async def get_documents():
    return list_documents()


@router.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    results = vector_search(req.query, top_k=req.top_k)
    return SearchResponse(
        query=req.query,
        results=[SearchResult(**r) for r in results],
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = answer(req.message)
    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        conversation_id=req.conversation_id or "",
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        answer_stream(req.message),
        media_type="text/event-stream",
    )
