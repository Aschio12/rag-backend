import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile

from app.config import settings
from app.ingestion import ingest_document
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.store import list_documents

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
