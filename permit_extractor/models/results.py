"""Top-level result objects: SheetResult and ExtractionResult.

ExtractionResult is the root output object serialised to JSON.
The rule engine attaches at ExtractionResult.checks — a list of Check objects
produced by either cross_checker.py or future rule modules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from permit_extractor.models.entities import ExtractedEntity
from permit_extractor.models.regions import Region, SheetInfo


@dataclass
class Check:
    """A single validation result.

    Produced by cross_checker.py today and by future rule modules later.
    Both use the same shape so the output format never needs to change.
    """
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    check_type: str = "cross_check"  # "cross_check" | "rule_<id>" in future
    severity: str = "warning"        # "info" | "warning" | "error"
    page_index: Optional[int] = None
    entity_ids: list[str] = field(default_factory=list)
    description: str = ""
    evidence: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "severity": self.severity,
            "page_index": self.page_index,
            "entity_ids": self.entity_ids,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class RunMetrics:
    """Timing and cost data for a single pipeline run."""
    total_pages: int = 0
    pages_with_text_layer: int = 0
    pages_raster: int = 0
    vlm_calls: int = 0
    vlm_prompt_tokens: int = 0
    vlm_completion_tokens: int = 0
    vlm_estimated_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SheetResult:
    """Extraction results for a single PDF page."""
    page_index: int
    sheet_info: SheetInfo
    regions: list[Region] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)

    # Sheet-level metadata resolved from title block
    sheet_number: str = "UNKNOWN"
    sheet_title: str = ""
    discipline: str = ""

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "sheet_number": self.sheet_number,
            "sheet_title": self.sheet_title,
            "discipline": self.discipline,
            "sheet_info": self.sheet_info.to_dict(),
            "regions": [r.to_dict() for r in self.regions],
            "entities": [e.to_dict() for e in self.entities],
        }


@dataclass
class ExtractionResult:
    """Root output object. Serialised to {stem}_extracted.json."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pdf_path: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config_snapshot: dict = field(default_factory=dict)

    sheets: list[SheetResult] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)  # rule engine seam
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def all_entities(self) -> list[ExtractedEntity]:
        return [e for s in self.sheets for e in s.entities]

    def entities_by_type(self, entity_type: str) -> list[ExtractedEntity]:
        return [e for e in self.all_entities() if e.entity_type == entity_type]

    def needs_review(self) -> list[ExtractedEntity]:
        """Entities with confidence < 0.7 or cross-check mismatch."""
        from permit_extractor.models.entities import CrossCheckStatus
        return [
            e for e in self.all_entities()
            if e.confidence < 0.7
            or e.cross_check_status == CrossCheckStatus.MISMATCH
        ]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "pdf_path": self.pdf_path,
            "timestamp": self.timestamp,
            "config_snapshot": self.config_snapshot,
            "sheets": [s.to_dict() for s in self.sheets],
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics.to_dict(),
        }
