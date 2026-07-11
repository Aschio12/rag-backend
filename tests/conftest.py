"""Pytest fixtures for RAG backend tests."""
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def temp_chroma_dir():
    """Use a temporary directory for ChromaDB/SQLite to avoid polluting the real one."""
    old = os.environ.get("CHROMA_PERSIST_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CHROMA_PERSIST_DIR"] = tmp
        yield tmp
    if old is not None:
        os.environ["CHROMA_PERSIST_DIR"] = old
    else:
        del os.environ["CHROMA_PERSIST_DIR"]
