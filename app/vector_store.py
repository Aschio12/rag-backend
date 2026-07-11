import logging
import re

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from app.config import settings

logger = logging.getLogger(__name__)

_embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=_embedding_function,
    metadata={"hnsw:space": "cosine"},
)

# BM25 index (built lazily)
_bm25_index = None
_bm25_docs = []
_bm25_id_map: dict[int, str] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _rebuild_bm25():
    """Rebuild BM25 index from ChromaDB contents."""
    global _bm25_index, _bm25_docs, _bm25_id_map
    try:
        all_data = _collection.get()
        if not all_data or not all_data.get("documents"):
            _bm25_index = None
            _bm25_docs = []
            _bm25_id_map = {}
            return

        from rank_bm25 import BM25Okapi

        _bm25_docs = list(all_data["documents"])
        tokenized = [_tokenize(d) for d in _bm25_docs]
        _bm25_index = BM25Okapi(tokenized)
        ids = list(all_data["ids"])
        _bm25_id_map = {i: ids[i] for i in range(len(ids))}
        logger.info(f"BM25 index rebuilt with {len(_bm25_docs)} documents")
    except Exception as e:
        logger.warning(f"BM25 rebuild error: {e}")
        _bm25_index = None


def add_chunks(doc_id: str, chunks: list[str], page_numbers: list[int] | None = None):
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        meta = {"doc_id": doc_id}
        if page_numbers and i < len(page_numbers):
            meta["page_number"] = page_numbers[i]
        metadatas.append(meta)
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    # Rebuild BM25 lazily
    global _bm25_index
    _bm25_index = None


def search(query: str, top_k: int = 5, hybrid: bool = False) -> list[dict]:
    """Search using vector similarity, optionally hybrid with BM25."""
    results = _collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "page_number": results["metadatas"][0][i].get("page_number", 0),
        })

    if hybrid:
        hits = _hybrid_fusion(hits, query, top_k)

    return hits


def _hybrid_fusion(vector_hits: list[dict], query: str, top_k: int) -> list[dict]:
    """Fuse vector search with BM25 using reciprocal rank fusion."""
    global _bm25_index
    if _bm25_index is None:
        _rebuild_bm25()
    if _bm25_index is None:
        return vector_hits

    # BM25 scores
    tokenized_query = _tokenize(query)
    bm25_scores = _bm25_index.get_scores(tokenized_query)

    # Create score dicts
    vector_scores: dict[str, float] = {}
    for i, hit in enumerate(vector_hits):
        vector_scores[hit["id"]] = 1.0 / (i + 1)

    bm25_scored: dict[str, float] = {}
    for idx, score in enumerate(bm25_scores):
        cid = _bm25_id_map.get(idx)
        if cid:
            bm25_scored[cid] = score

    # Reciprocal rank fusion
    fused: dict[str, float] = {}
    all_ids = set(list(vector_scores.keys()) + list(bm25_scored.keys()))
    for cid in all_ids:
        vs = vector_scores.get(cid, 0.0)
        bs = bm25_scored.get(cid, 0.0)
        fused[cid] = vs + bs * 0.3  # Weighted fusion

    sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]

    # Map back to hit data
    hit_map = {h["id"]: h for h in vector_hits}
    result = []
    for cid in sorted_ids:
        if cid in hit_map:
            h = dict(hit_map[cid])
            h["score"] = fused[cid]
            h["hybrid_score"] = True
            result.append(h)
        else:
            # From BM25 only - need to get data from ChromaDB
            try:
                data = _collection.get(ids=[cid])
                if data and data["documents"]:
                    result.append({
                        "id": cid,
                        "text": data["documents"][0],
                        "score": fused[cid],
                        "doc_id": data["metadatas"][0].get("doc_id", "") if data.get("metadatas") else "",
                        "page_number": data["metadatas"][0].get("page_number", 0) if data.get("metadatas") else 0,
                        "hybrid_score": True,
                    })
            except Exception:
                pass

    return result


def delete_document(doc_id: str):
    _collection.delete(where={"doc_id": doc_id})
    global _bm25_index
    _bm25_index = None


def get_chunks_for_doc(doc_id: str) -> list[dict]:
    """Get all chunks for a document."""
    try:
        results = _collection.get(where={"doc_id": doc_id})
        if not results or not results.get("ids"):
            return []
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i] if results.get("documents") else "",
                "page_number": results["metadatas"][i].get("page_number", 0) if results.get("metadatas") else 0,
            })
        return chunks
    except Exception:
        return []
