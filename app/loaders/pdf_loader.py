import fitz

from app.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
