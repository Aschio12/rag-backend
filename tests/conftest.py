"""Pytest fixtures for RAG backend tests."""
import importlib
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def temp_dirs(monkeypatch):
    """Use temporary directories for ChromaDB and SQLite to avoid polluting real data."""
    tmp = tempfile.mkdtemp()

    # Patch the settings
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", tmp)

    # Reload dependent modules so they pick up the patched path
    for mod_name in ["app.database", "app.vector_store"]:
        if mod_name in importlib.import_module("sys").modules:
            importlib.reload(importlib.import_module(mod_name))

    yield

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
