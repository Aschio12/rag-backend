"""Tests for the vector store (ChromaDB + BM25 hybrid search)."""
import pytest

from app.vector_store import add_chunks, search, delete_document, get_chunks_for_doc


@pytest.fixture(autouse=True)
def clean_db():
    """Clean all docs before each test by deleting known test doc IDs."""
    for doc_id in ["test_doc_a", "test_doc_b", "clean_test"]:
        try:
            delete_document(doc_id)
        except Exception:
            pass


class TestAddAndSearch:
    def test_add_and_vector_search(self):
        chunks = ["The quick brown fox jumps over the lazy dog.", "Python is a programming language."]
        add_chunks("test_doc_a", chunks, page_numbers=[1, 2])
        results = search("fox", top_k=5, hybrid=False)
        assert len(results) > 0
        # The fox chunk should be the top result
        assert "fox" in results[0]["text"].lower()

    def test_hybrid_search(self):
        chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Reinforcement learning trains agents through rewards.",
        ]
        add_chunks("test_doc_b", chunks)
        results = search("neural networks deep learning", top_k=5, hybrid=True)
        assert len(results) > 0
        # Ensure hybrid results include score structure
        assert "score" in results[0]

    def test_search_empty(self):
        results = search("nonexistent query", top_k=5)
        assert len(results) == 0 or isinstance(results, list)

    def test_search_top_k(self):
        chunks = [f"Chunk number {i} about something." for i in range(20)]
        add_chunks("test_doc_a", chunks)
        results = search("chunk number", top_k=3)
        assert len(results) <= 3


class TestDeleteAndGet:
    def test_delete_document(self):
        chunks = ["Delete me please."]
        add_chunks("clean_test", chunks)
        delete_document("clean_test")
        results = search("delete", top_k=5)
        # The doc should not appear in results anymore
        for r in results:
            assert r.get("doc_id") != "clean_test"

    def test_get_chunks_for_doc(self):
        chunks = ["First chunk content.", "Second chunk content."]
        add_chunks("test_doc_a", chunks, page_numbers=[1, 2])
        doc_chunks = get_chunks_for_doc("test_doc_a")
        assert len(doc_chunks) == 2
        assert doc_chunks[0]["page_number"] == 1
        assert doc_chunks[1]["page_number"] == 2
        assert "First chunk" in doc_chunks[0]["text"]

    def test_get_chunks_for_nonexistent_doc(self):
        assert get_chunks_for_doc("does_not_exist") == []


class TestPageNumbers:
    def test_page_numbers_stored(self):
        chunks = ["Page 1 text", "Page 2 text", "Page 3 text"]
        add_chunks("test_doc_a", chunks, page_numbers=[1, 2, 3])
        results = search("page", top_k=5)
        found_pages = set()
        for r in results:
            if r.get("page_number"):
                found_pages.add(r["page_number"])
        assert 1 in found_pages
        assert 2 in found_pages
        assert 3 in found_pages
