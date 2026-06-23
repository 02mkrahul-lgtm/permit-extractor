"""Local Tesseract OCR provider.

Requires tesseract binary on PATH: brew install tesseract (macOS)
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from permit_extractor.providers.base import OCRProvider

logger = logging.getLogger(__name__)

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


class TesseractOCRProvider(OCRProvider):
    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        if not _TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image: Image.Image) -> list[dict]:
        """Return word-level OCR results with per-word confidence and bbox."""
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # assume uniform block of text
            )
        except Exception as exc:
            logger.error("Tesseract failed: %s", exc)
            return []

        results = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf_raw = data["conf"][i]
            if not text or conf_raw < 0:
                continue
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            results.append({
                "text": text,
                "confidence": round(conf_raw / 100.0, 3),
                "bbox": {"x0": float(x), "y0": float(y), "x1": float(x + w), "y1": float(y + h)},
            })
        return results
