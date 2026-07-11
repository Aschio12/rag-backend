import re

from app.config import settings


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    step = size - overlap

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += step

    return [c for c in chunks if c]


def chunk_text_with_pages(
    text: str,
    page_map: dict[int, str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[list[str], list[int]]:
    """Chunk text and track page numbers for each chunk."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    step = size - overlap

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return [], []

    # Build position-to-page mapping
    pos_to_page: dict[int, int] = {}
    if page_map:
        sorted_pages = sorted(page_map.keys())
        for i in range(len(sorted_pages)):
            start_pos = sorted_pages[i]
            end_pos = sorted_pages[i + 1] if i + 1 < len(sorted_pages) else len(text)
            for pos in range(start_pos, end_pos):
                pos_to_page[pos] = i + 1

    chunks = []
    pages = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            # Determine page number for this chunk
            if pos_to_page:
                page = pos_to_page.get(start, 0)
                pages.append(page)
            else:
                pages.append(0)
        start += step

    return chunks, pages
