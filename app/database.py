import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings

DB_PATH = Path(settings.chroma_persist_dir) / "metadata.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL DEFAULT '',
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            error TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chroma_id TEXT NOT NULL,
            text TEXT NOT NULL,
            page_number INTEGER DEFAULT 0,
            bbox TEXT DEFAULT '',
            score REAL DEFAULT 0.0,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_collections_kb ON collections(kb_id);
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    """)
    conn.commit()
    conn.close()


# ---- Knowledge Bases ----
def create_kb(name: str, description: str = "") -> dict:
    conn = _get_conn()
    kb_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO knowledge_bases (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (kb_id, name, description, now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    conn.close()
    return dict(row)


def list_kbs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kb(kb_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_kb(kb_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[dict]:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    updates = []
    params = []
    if name is not None:
        updates.append("name=?")
        params.append(name)
    if description is not None:
        updates.append("description=?")
        params.append(description)
    if updates:
        updates.append("updated_at=?")
        params.append(now)
        params.append(kb_id)
        conn.execute(f"UPDATE knowledge_bases SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_kb(kb_id: str) -> bool:
    conn = _get_conn()
    conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
    affected = conn.total_changes
    conn.commit()
    conn.close()
    return affected > 0


# ---- Collections ----
def create_collection(kb_id: str, name: str, description: str = "") -> dict:
    conn = _get_conn()
    col_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO collections (id, kb_id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (col_id, kb_id, name, description, now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM collections WHERE id=?", (col_id,)).fetchone()
    conn.close()
    return dict(row)


def list_collections(kb_id: Optional[str] = None) -> list[dict]:
    conn = _get_conn()
    if kb_id:
        rows = conn.execute(
            "SELECT * FROM collections WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_collection(col_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM collections WHERE id=?", (col_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_collection(col_id: str) -> bool:
    conn = _get_conn()
    conn.execute("DELETE FROM collections WHERE id=?", (col_id,))
    affected = conn.total_changes
    conn.commit()
    conn.close()
    return affected > 0


# ---- Documents ----
def add_document_meta(
    doc_id: str,
    collection_id: str,
    filename: str,
    file_path: str,
    file_type: str = "",
    page_count: int = 0,
    chunk_count: int = 0,
    status: str = "pending",
    metadata: dict | None = None,
) -> dict:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO documents
           (id, collection_id, filename, file_path, file_type, page_count, chunk_count, status, metadata, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, collection_id, filename, file_path, file_type, page_count, chunk_count, status, json.dumps(metadata or {}), now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row)


def list_documents_meta(collection_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM documents WHERE 1=1"
    params = []
    if collection_id:
        query += " AND collection_id=?"
        params.append(collection_id)
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document_meta(doc_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_document_status(doc_id: str, status: str, error: str = "", chunk_count: Optional[int] = None):
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    updates = ["status=?", "updated_at=?"]
    params = [status, now]
    if error:
        updates.append("error=?")
        params.append(error)
    if chunk_count is not None:
        updates.append("chunk_count=?")
        params.append(chunk_count)
    params.append(doc_id)
    conn.execute(f"UPDATE documents SET {', '.join(updates)} WHERE id=?", params)
    conn.commit()
    conn.close()


def delete_document_meta(doc_id: str) -> bool:
    conn = _get_conn()
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    affected = conn.total_changes
    conn.commit()
    conn.close()
    return affected > 0
