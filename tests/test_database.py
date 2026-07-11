"""Tests for the SQLite-based metadata database layer."""
import pytest

from app.database import (
    init_db,
    create_kb,
    list_kbs,
    get_kb,
    update_kb,
    delete_kb,
    create_collection,
    list_collections,
    get_collection,
    delete_collection,
    add_document_meta,
    list_documents_meta,
    get_document_meta,
    delete_document_meta,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


class TestKnowledgeBases:
    def test_create_and_list(self):
        kb = create_kb("Test KB", "A test")
        assert kb["name"] == "Test KB"
        assert kb["description"] == "A test"
        kbs = list_kbs()
        assert any(k["id"] == kb["id"] for k in kbs)

    def test_get_kb(self):
        kb = create_kb("Get Test")
        fetched = get_kb(kb["id"])
        assert fetched is not None
        assert fetched["name"] == "Get Test"

    def test_get_kb_not_found(self):
        assert get_kb("nonexistent") is None

    def test_update_kb(self):
        kb = create_kb("Old Name")
        updated = update_kb(kb["id"], name="New Name")
        assert updated["name"] == "New Name"
        assert updated["description"] == ""

    def test_delete_kb(self):
        kb = create_kb("To Delete")
        assert delete_kb(kb["id"]) is True
        assert get_kb(kb["id"]) is None

    def test_delete_kb_not_found(self):
        assert delete_kb("nonexistent") is False


class TestCollections:
    def test_create_and_list(self):
        kb = create_kb("KB for Collections")
        col = create_collection(kb["id"], "Col1", "A collection")
        assert col["name"] == "Col1"
        cols = list_collections(kb["id"])
        assert any(c["id"] == col["id"] for c in cols)

    def test_list_all_collections(self):
        kb = create_kb("KB All")
        create_collection(kb["id"], "C1")
        create_collection(kb["id"], "C2")
        cols = list_collections()
        assert len([c for c in cols if c["kb_id"] == kb["id"]]) >= 2

    def test_get_collection(self):
        kb = create_kb("KB Get Col")
        col = create_collection(kb["id"], "Get Me")
        fetched = get_collection(col["id"])
        assert fetched["name"] == "Get Me"

    def test_delete_collection(self):
        kb = create_kb("KB Del Col")
        col = create_collection(kb["id"], "Delete Me")
        assert delete_collection(col["id"]) is True
        assert get_collection(col["id"]) is None

    def test_cascade_on_kb_delete(self):
        """Deleting a KB should cascade delete its collections."""
        kb = create_kb("KB Cascade")
        col = create_collection(kb["id"], "Cascade Col")
        delete_kb(kb["id"])
        assert get_collection(col["id"]) is None


class TestDocuments:
    def test_add_and_list(self):
        kb = create_kb("KB Docs")
        col = create_collection(kb["id"], "Col Docs")
        doc = add_document_meta(
            doc_id="doc1",
            collection_id=col["id"],
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_type="pdf",
            page_count=5,
            status="ready",
        )
        assert doc["filename"] == "test.pdf"
        docs = list_documents_meta(collection_id=col["id"])
        assert any(d["id"] == "doc1" for d in docs)

    def test_list_filter_by_status(self):
        kb = create_kb("KB Status")
        col = create_collection(kb["id"], "Col Status")
        add_document_meta("d1", col["id"], "a.pdf", "/tmp/a.pdf", status="ready")
        add_document_meta("d2", col["id"], "b.pdf", "/tmp/b.pdf", status="pending")
        ready_docs = list_documents_meta(status="ready")
        assert any(d["id"] == "d1" for d in ready_docs)
        assert not any(d["id"] == "d2" for d in ready_docs)

    def test_get_document_meta(self):
        kb = create_kb("KB Get Doc")
        col = create_collection(kb["id"], "Col Get Doc")
        add_document_meta("get1", col["id"], "doc.pdf", "/tmp/doc.pdf")
        doc = get_document_meta("get1")
        assert doc["filename"] == "doc.pdf"

    def test_delete_document_meta(self):
        kb = create_kb("KB Del Doc")
        col = create_collection(kb["id"], "Col Del Doc")
        add_document_meta("del1", col["id"], "del.pdf", "/tmp/del.pdf")
        assert delete_document_meta("del1") is True
        assert get_document_meta("del1") is None

    def test_cascade_on_collection_delete(self):
        """Deleting a collection should cascade delete its docs."""
        kb = create_kb("KB Cascade Doc")
        col = create_collection(kb["id"], "Col Cascade")
        add_document_meta("c1", col["id"], "c.pdf", "/tmp/c.pdf")
        delete_collection(col["id"])
        assert get_document_meta("c1") is None
