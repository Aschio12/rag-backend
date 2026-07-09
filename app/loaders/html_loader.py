from pathlib import Path

from bs4 import BeautifulSoup

from app.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        html = Path(file_path).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n").strip()
