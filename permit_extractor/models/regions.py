"""Region types, BoundingBox, and per-page sheet metadata.

BoundingBox lives here (not in entities.py) so that entities.py can import
from this module without creating a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class BoundingBox:
    """Coordinates in PDF points (1/72 inch) relative to the page origin (bottom-left)."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_index: int  # 0-based

    def to_dict(self) -> dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page_index": self.page_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(**d)

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0


class RegionType(str, Enum):
    TITLE_BLOCK = "title_block"
    SCHEDULE = "schedule"
    NOTES = "notes"
    LEGEND = "legend"
    DRAWING_BODY = "drawing_body"


@dataclass
class Region:
    """A detected layout region on a single sheet."""
    region_type: RegionType
    bbox: BoundingBox       # in PDF points
    bbox_px: Optional[BoundingBox] = None  # in pixels at render DPI (for image crops)
    label: Optional[str] = None  # e.g. "Door Schedule", "General Notes"

    def to_dict(self) -> dict:
        return {
            "region_type": self.region_type.value,
            "bbox": self.bbox.to_dict(),
            "bbox_px": self.bbox_px.to_dict() if self.bbox_px else None,
            "label": self.label,
        }


@dataclass
class SheetInfo:
    """Per-page metadata determined during ingestion."""
    page_index: int
    has_text_layer: bool        # True → prefer vector extraction
    text_layer_coverage: float  # 0.0–1.0: fraction of expected text found via get_text
    width_pt: float             # page width in PDF points
    height_pt: float
    render_dpi: int
    image_width_px: int
    image_height_px: int

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "has_text_layer": self.has_text_layer,
            "text_layer_coverage": self.text_layer_coverage,
            "width_pt": self.width_pt,
            "height_pt": self.height_pt,
            "render_dpi": self.render_dpi,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
        }
