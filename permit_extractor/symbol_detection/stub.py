"""Symbol detection stub — interface only.

NOT implemented in v1. The interface is defined here so that:
  1. The pipeline can reference it without breaking.
  2. A future implementation can slot in without touching pipeline.py.

To implement: replace SymbolDetector with a concrete subclass using
template matching, a fine-tuned detection model, or geometric analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class DetectedSymbol:
    symbol_type: str          # "door", "window", "fixture", "column", "exit_sign", etc.
    bbox_px: tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels
    confidence: float
    attributes: Optional[dict] = None


class SymbolDetector(ABC):
    """Abstract interface for construction drawing symbol detection."""

    @abstractmethod
    def detect(self, image: Image.Image, page_index: int) -> list[DetectedSymbol]:
        """Detect symbols in a full-sheet image.

        Returns a list of DetectedSymbol objects.
        Not called in v1 — the pipeline skips this stage.
        """
        ...


class NotImplementedSymbolDetector(SymbolDetector):
    """Placeholder returned when symbol detection is not yet configured."""

    def detect(self, image: Image.Image, page_index: int) -> list[DetectedSymbol]:
        return []  # no-op stub
