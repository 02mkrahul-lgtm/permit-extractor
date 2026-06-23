"""Core entity model — every extracted fact lives here.

All fields are mandatory provenance for the downstream rule engine.
Do NOT remove fields from ExtractedEntity without updating the rule engine seam.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from permit_extractor.models.regions import BoundingBox, RegionType


class ExtractionMethod(str, Enum):
    VECTOR = "vector"   # text layer present; extracted via PyMuPDF/pdfplumber
    OCR = "ocr"         # raster page; extracted via Tesseract
    VLM = "vlm"         # vision-language model on a cropped region image


class CrossCheckStatus(str, Enum):
    MATCHED = "matched"
    MISMATCH = "mismatch"
    VLM_ONLY = "vlm_only"
    VECTOR_ONLY = "vector_only"


@dataclass
class ExtractedEntity:
    """A single extracted fact with full provenance.

    This is the unit of data that flows through every pipeline stage and that
    the rule engine will consume. The provenance fields (sheet_number through
    raw_source) are MANDATORY — never emit an entity without them.
    """
    # --- Value ----------------------------------------------------------
    entity_type: str    # e.g. "sheet_number", "door_schedule_row", "occupancy_class"
    value: Any          # structured value (str, dict, list, numeric)

    # --- Provenance (mandatory) -----------------------------------------
    sheet_number: str           # resolved from title block; "UNKNOWN" until resolved
    page_index: int             # 0-based PDF page index
    region_type: RegionType
    bbox: BoundingBox
    extraction_method: ExtractionMethod
    confidence: float           # 0.0–1.0; calibrated by feedback/confidence_calibrator.py
    raw_source: Optional[str] = None  # text/OCR bytes before semantic parsing

    # --- Cross-check (populated by validation/cross_checker.py) ---------
    cross_check_status: Optional[CrossCheckStatus] = None
    cross_check_delta: Optional[dict] = None  # what differed

    # --- Feedback linkage (populated after correction ingestion) --------
    correction_id: Optional[str] = None

    # --- Internal ID ----------------------------------------------------
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "value": self.value,
            "sheet_number": self.sheet_number,
            "page_index": self.page_index,
            "region_type": self.region_type.value if hasattr(self.region_type, "value") else self.region_type,
            "bbox": self.bbox.to_dict(),
            "extraction_method": self.extraction_method.value,
            "confidence": self.confidence,
            "raw_source": self.raw_source,
            "cross_check_status": self.cross_check_status.value if self.cross_check_status else None,
            "cross_check_delta": self.cross_check_delta,
            "correction_id": self.correction_id,
        }
