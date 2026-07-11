import logging

import fitz

from app.loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """PDF loader with page-tracking and optional OCR fallback."""

    def load(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text = page.get_text().strip()
            if not text:
                text = self._ocr_page(page)
            text_parts.append(text)
        doc.close()
        return "\n\n".join(text_parts)

    def load_with_pages(self, file_path: str) -> tuple[str, dict[int, str], int]:
        """Returns (full_text, page_map, page_count).
        page_map: page_number -> text for that page
        """
        doc = fitz.open(file_path)
        page_map: dict[int, str] = {}
        text_parts = []
        pos = 0
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                text = self._ocr_page(page)
            page_map[pos] = text
            text_parts.append(text)
            pos += len(text) + 2  # +2 for the \n\n separator
        doc.close()
        return "\n\n".join(text_parts), page_map, len(doc)

    def _ocr_page(self, page) -> str:
        """Fallback OCR using pytesseract if available."""
        try:
            import pytesseract
            from PIL import Image
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            logger.info(f"OCR applied to page {page.number + 1}")
            return text.strip()
        except ImportError:
            logger.warning(f"pytesseract not available for OCR on page {page.number + 1}")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed on page {page.number + 1}: {e}")
            return ""

    def get_preview_images(self, file_path: str, max_pages: int = 5) -> list[bytes]:
        """Convert PDF pages to PNG images for preview."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path, first_page=1, last_page=max_pages)
            from io import BytesIO
            previews = []
            for img in images:
                buf = BytesIO()
                img.save(buf, format="PNG")
                previews.append(buf.getvalue())
            return previews
        except Exception as e:
            logger.error(f"Preview generation error: {e}")
            return []
