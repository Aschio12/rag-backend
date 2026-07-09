import uuid
from pathlib import Path

from app.chunker import chunk_text
from app.loaders import load_document
from app.store import add_document


def ingest_document(file_path: str) -> dict:
    doc_id = str(uuid.uuid4())
    filename = Path(file_path).name

    text = load_document(file_path)
    chunks = chunk_text(text)

    doc = {
        "id": doc_id,
        "filename": filename,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": "indexed",
    }
    add_document(doc)
    return doc
