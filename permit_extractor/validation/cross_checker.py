"""Cross-check vector/OCR entities against VLM entities for the same region.

Produces Check objects (defined in models/results.py) — the same type that
future rule modules will also produce. This is the seam for the rule engine.

When both a vector/OCR entity and a VLM entity cover the same region and
entity type, we compare their values and flag mismatches.
"""

from __future__ import annotations

import logging
from typing import Optional

from permit_extractor.models.entities import CrossCheckStatus, ExtractedEntity, ExtractionMethod
from permit_extractor.models.results import Check

logger = logging.getLogger(__name__)

_HIGH_FIDELITY = {ExtractionMethod.VECTOR, ExtractionMethod.OCR}


def cross_check_entities(entities: list[ExtractedEntity]) -> tuple[list[ExtractedEntity], list[Check]]:
    """Compare vector/OCR vs VLM entities for the same (region_type, entity_type) pairs.

    Returns:
        (updated_entities, checks)
        - updated_entities: same list with cross_check_status populated
        - checks: Check objects for mismatches (consumable by downstream rule engine)
    """
    checks: list[Check] = []

    # Group by (page_index, region_type, base_entity_type)
    # "base" strips the region prefix to normalise across methods
    groups: dict[tuple, list[ExtractedEntity]] = {}
    for e in entities:
        key = (e.page_index, e.region_type, _base_type(e.entity_type))
        groups.setdefault(key, []).append(e)

    for key, group in groups.items():
        hf = [e for e in group if e.extraction_method in _HIGH_FIDELITY]
        vlm = [e for e in group if e.extraction_method == ExtractionMethod.VLM]

        if hf and vlm:
            _reconcile(hf, vlm, checks)
        elif hf and not vlm:
            for e in hf:
                e.cross_check_status = CrossCheckStatus.VECTOR_ONLY
        elif vlm and not hf:
            for e in vlm:
                e.cross_check_status = CrossCheckStatus.VLM_ONLY

    return entities, checks


def _reconcile(
    hf_entities: list[ExtractedEntity],
    vlm_entities: list[ExtractedEntity],
    checks: list[Check],
) -> None:
    """Compare high-fidelity vs VLM entities and populate cross_check_status."""
    hf_text = _entity_text(hf_entities[0])
    vlm_text = _entity_text(vlm_entities[0])

    if hf_text and vlm_text:
        similarity = _text_similarity(hf_text, vlm_text)
        if similarity >= 0.85:
            for e in hf_entities + vlm_entities:
                e.cross_check_status = CrossCheckStatus.MATCHED
        else:
            delta = {
                "vector_value": hf_text[:500],
                "vlm_value": vlm_text[:500],
                "similarity": round(similarity, 3),
            }
            for e in hf_entities + vlm_entities:
                e.cross_check_status = CrossCheckStatus.MISMATCH
                e.cross_check_delta = delta

            checks.append(Check(
                check_type="cross_check",
                severity="warning",
                page_index=hf_entities[0].page_index,
                entity_ids=[e.entity_id for e in hf_entities + vlm_entities],
                description=(
                    f"Vector/VLM mismatch in {hf_entities[0].region_type.value} "
                    f"(entity: {_base_type(hf_entities[0].entity_type)}): "
                    f"similarity={similarity:.2f}"
                ),
                evidence=delta,
            ))
    else:
        for e in hf_entities + vlm_entities:
            e.cross_check_status = CrossCheckStatus.MATCHED


def _entity_text(entity: ExtractedEntity) -> str:
    """Get a flat string representation of an entity's value for comparison."""
    v = entity.value
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, dict):
        return " ".join(f"{k}:{val}" for k, val in sorted(v.items(), key=lambda x: str(x[0])))
    return str(v).strip().lower()


def _text_similarity(a: str, b: str) -> float:
    """Simple character-level Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Token-based Jaccard
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _base_type(entity_type: str) -> str:
    """Strip region prefix to get a normalised type key for grouping."""
    for prefix in ("title_block_text", "schedule_text", "notes_text",
                   "title_", "schedule_", "notes_"):
        if entity_type.startswith(prefix):
            return entity_type[len(prefix):]
    return entity_type
