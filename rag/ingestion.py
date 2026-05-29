import fitz  # pymupdf
import re
from typing import List, Dict


def _clean(text: str) -> str:
    """Remove control characters, normalize whitespace."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract and clean text from a PDF, one dict per page."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = _clean(page.get_text())
        if len(text) > 50:
            pages.append({"text": text, "page": i + 1, "source": filename})
    doc.close()
    return pages


def chunk_text(pages: List[Dict], chunk_size: int = 200, overlap: int = 20) -> List[Dict]:
    """Split pages into small overlapping word-based chunks."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + chunk_size]
            if len(chunk_words) < 15:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page["page"],
                "source": page["source"],
                "chunk_id": len(chunks),
            })
    return chunks
