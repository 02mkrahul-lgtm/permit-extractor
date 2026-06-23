"""Stream pages from a PDF without loading the full file into memory.

Yields (page_index, fitz.Page) one page at a time so large permit sets
don't exhaust RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import fitz  # PyMuPDF


def open_pdf(pdf_path: str | Path) -> fitz.Document:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(str(path))
    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {path}")
    return doc


def iter_pages(pdf_path: str | Path) -> Generator[Tuple[int, fitz.Page], None, None]:
    """Yield (0-based page_index, fitz.Page) without caching all pages.

    The caller must NOT store the Page object beyond each loop iteration;
    fitz invalidates pages when the document is closed.
    """
    doc = open_pdf(pdf_path)
    try:
        for idx in range(doc.page_count):
            yield idx, doc.load_page(idx)
    finally:
        doc.close()


def page_count(pdf_path: str | Path) -> int:
    doc = open_pdf(pdf_path)
    n = doc.page_count
    doc.close()
    return n
