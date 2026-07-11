import uuid
from pathlib import Path

from app.chunker import chunk_text, chunk_text_with_pages
from app.loaders import load_document
from app.store import add_document
from app.vector_store import add_chunks


def ingest_document(file_path: str) -> dict:
    doc_id = str(uuid.uuid4())
    filename = Path(file_path).name

    text = load_document(file_path)
    chunks = chunk_text(text)

    add_chunks(doc_id, chunks)

    doc = {
        "id": doc_id,
        "filename": filename,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": "indexed",
    }
    add_document(doc)
    return doc


def ingest_document_with_pages(file_path: str, page_map: dict[int, str] | None = None) -> dict:
    """Ingest with page number tracking."""
    doc_id = str(uuid.uuid4())
    filename = Path(file_path).name

    text = load_document(file_path)
    chunks, pages = chunk_text_with_pages(text, page_map=page_map)

    add_chunks(doc_id, chunks, page_numbers=pages)

    doc = {
        "id": doc_id,
        "filename": filename,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": "indexed",
    }
    add_document(doc)
    return doc
