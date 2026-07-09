from typing import Any

_documents: dict[str, dict[str, Any]] = {}


def add_document(doc: dict[str, Any]):
    _documents[doc["id"]] = doc


def get_document(doc_id: str) -> dict[str, Any] | None:
    return _documents.get(doc_id)


def list_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": d["id"],
            "filename": d["filename"],
            "chunk_count": d["chunk_count"],
            "status": d.get("status", "indexed"),
        }
        for d in _documents.values()
    ]


def remove_document(doc_id: str):
    _documents.pop(doc_id, None)
