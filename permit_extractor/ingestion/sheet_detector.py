"""Detect per-page properties: text layer vs. raster, and render to PIL Image.

Text-layer detection:
  - Use fitz.Page.get_text() char count as the primary signal.
  - A page is considered to have a text layer when it has ≥ threshold chars.
  - Record coverage (ratio) so downstream stages can make nuanced decisions.
"""

from __future__ import annotations

import io
from typing import Tuple

import fitz  # PyMuPDF
from PIL import Image

from permit_extractor.models.regions import SheetInfo


def detect_and_render(
    page: fitz.Page,
    page_index: int,
    dpi: int = 300,
    char_threshold: int = 50,
) -> Tuple[SheetInfo, Image.Image]:
    """Return (SheetInfo, PIL Image rendered at `dpi`).

    The image is in RGB mode, ready for OpenCV (after converting to numpy)
    and for saving as JPEG/PNG.
    """
    # --- Text layer detection ------------------------------------------
    text = page.get_text("text")
    char_count = len(text.strip())
    has_text_layer = char_count >= char_threshold

    # Coverage: normalise by expected chars for an A1 sheet at 300 DPI.
    # This is a heuristic — it only needs to be meaningful, not precise.
    expected_chars = max(char_threshold * 10, 500)
    coverage = min(1.0, char_count / expected_chars)

    rect = page.rect
    info = SheetInfo(
        page_index=page_index,
        has_text_layer=has_text_layer,
        text_layer_coverage=round(coverage, 3),
        width_pt=rect.width,
        height_pt=rect.height,
        render_dpi=dpi,
        image_width_px=0,   # filled below
        image_height_px=0,
    )

    # --- Render page to image ------------------------------------------
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    info.image_width_px = pix.width
    info.image_height_px = pix.height

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return info, img
