"""Write ExtractionResult to {stem}_extracted.json."""

from __future__ import annotations

import json
from pathlib import Path

from permit_extractor.models.results import ExtractionResult


def write_json(result: ExtractionResult, output_dir: str, stem: str) -> Path:
    out = Path(output_dir) / f"{stem}_extracted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    return out
