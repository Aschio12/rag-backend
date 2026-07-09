import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from app.config import settings

_embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=_embedding_function,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(doc_id: str, chunks: list[str]):
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id} for _ in chunks]
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)


def search(query: str, top_k: int = 5) -> list[dict]:
    results = _collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
        })
    return hits


def delete_document(doc_id: str):
    _collection.delete(where={"doc_id": doc_id})
