"""Correction and feedback data models.

These are the structures that flow through the review → feedback loop.
Keep them exportable as JSON so they can seed fine-tuning later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class CorrectionStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    MISSING = "missing"     # entity should have been extracted but wasn't
    PENDING = "pending"     # not yet reviewed


@dataclass
class CorrectionRecord:
    """A single human correction on one extracted entity.

    Pre-populated by review_file.py; filled in by the reviewer;
    ingested by `feedback apply` → stored in SQLite.
    """
    correction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""

    # Entity provenance (copied from ExtractedEntity at review time)
    entity_id: str = ""
    entity_type: str = ""
    page_index: int = 0
    region_type: str = ""
    sheet_number: str = ""
    extraction_method: str = ""
    predicted_confidence: float = 0.0

    # Values
    predicted_value: Any = None
    corrected_value: Any = None     # None if status == correct
    status: CorrectionStatus = CorrectionStatus.PENDING

    # Context saved for exemplar/fine-tuning use
    region_crop_path: Optional[str] = None   # path to saved crop image
    raw_source: Optional[str] = None

    # Metadata
    reviewer_note: Optional[str] = None
    reviewed_at: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "correction_id": self.correction_id,
            "run_id": self.run_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "page_index": self.page_index,
            "region_type": self.region_type,
            "sheet_number": self.sheet_number,
            "extraction_method": self.extraction_method,
            "predicted_confidence": self.predicted_confidence,
            "predicted_value": self.predicted_value,
            "corrected_value": self.corrected_value,
            "status": self.status.value,
            "region_crop_path": self.region_crop_path,
            "raw_source": self.raw_source,
            "reviewer_note": self.reviewer_note,
            "reviewed_at": self.reviewed_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorrectionRecord":
        d = d.copy()
        if "status" in d:
            d["status"] = CorrectionStatus(d["status"])
        return cls(**d)


@dataclass
class FeedbackEntry:
    """A confirmed correction stored in the feedback SQLite store.

    Distinct from CorrectionRecord: this is the persisted, aggregated form
    used by the exemplar retriever and failure-pattern catalog.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correction_id: str = ""
    entity_type: str = ""
    region_type: str = ""
    sheet_discipline: str = ""  # e.g. "ARCHITECTURAL", "ELECTRICAL"
    extraction_method: str = ""
    predicted_value: Any = None
    corrected_value: Any = None
    region_crop_path: Optional[str] = None
    raw_source: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
