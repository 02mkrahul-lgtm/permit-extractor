"""Generate the pre-populated review file for human correction.

Output: {stem}_review.json — sorted by confidence ascending so the
reviewer's attention goes to the weakest extractions first.

The reviewer fills in `status` (correct/incorrect/missing) and
`corrected_value` for each entry, then runs:
    permit-extractor feedback apply {stem}_review.json
"""

from __future__ import annotations

import json
from pathlib import Path

from permit_extractor.models.corrections import CorrectionRecord, CorrectionStatus
from permit_extractor.models.results import ExtractionResult


def write_review_file(result: ExtractionResult, output_dir: str, stem: str) -> Path:
    out = Path(output_dir) / f"{stem}_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    records = _build_records(result)
    # Sort: lowest confidence first, then mismatches, then alphabetically
    records.sort(key=lambda r: (
        r.predicted_confidence,
        r.status != CorrectionStatus.PENDING,
    ))

    payload = {
        "run_id": result.run_id,
        "pdf_path": result.pdf_path,
        "instructions": (
            "For each item: set 'status' to 'correct', 'incorrect', or 'missing'. "
            "If 'incorrect', supply 'corrected_value'. "
            "Run: permit-extractor feedback apply <this_file> when done."
        ),
        "items": [r.to_dict() for r in records],
    }

    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    return out


def _build_records(result: ExtractionResult) -> list[CorrectionRecord]:
    records = []
    for entity in result.all_entities():
        records.append(CorrectionRecord(
            run_id=result.run_id,
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            page_index=entity.page_index,
            region_type=entity.region_type.value if hasattr(entity.region_type, "value") else str(entity.region_type),
            sheet_number=entity.sheet_number,
            extraction_method=entity.extraction_method.value,
            predicted_confidence=entity.confidence,
            predicted_value=entity.value,
            corrected_value=None,
            status=CorrectionStatus.PENDING,
            raw_source=entity.raw_source,
        ))
    return records
