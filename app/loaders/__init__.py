from pathlib import Path

from app.loaders.html_loader import HTMLLoader
from app.loaders.markdown_loader import MarkdownLoader
from app.loaders.pdf_loader import PDFLoader

_loaders = {
    ".pdf": PDFLoader(),
    ".md": MarkdownLoader(),
    ".mdx": MarkdownLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loader = _loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader


def load_document(file_path: str) -> str:
    loader = get_loader(file_path)
    return loader.load(file_path)


__all__ = ["PDFLoader", "MarkdownLoader", "HTMLLoader", "get_loader", "load_document"]
