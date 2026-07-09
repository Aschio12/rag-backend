from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from app.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        text = Path(file_path).read_text(encoding="utf-8")
        md = MarkdownIt()
        html = md.render(text)
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n").strip()
