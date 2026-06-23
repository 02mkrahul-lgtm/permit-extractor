"""OCR fallback extraction for raster/scanned pages.

Used when a page has no text layer (sheet_info.has_text_layer is False).
Runs per-region crops through the OCRProvider and emits text entities
tagged extraction_method=OCR.
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from permit_extractor.models.entities import ExtractedEntity, ExtractionMethod
from permit_extractor.models.regions import BoundingBox
from permit_extractor.models.regions import Region, RegionType
from permit_extractor.providers.base import OCRProvider
from permit_extractor.segmentation.layout_segmenter import crop_region

logger = logging.getLogger(__name__)


def extract_ocr_entities(
    image: Image.Image,
    regions: list[Region],
    ocr_provider: OCRProvider,
    page_index: int,
    sheet_number: str = "UNKNOWN",
) -> list[ExtractedEntity]:
    """Run OCR on each region crop and return text entities."""
    entities: list[ExtractedEntity] = []

    for region in regions:
        if region.region_type == RegionType.DRAWING_BODY:
            continue

        try:
            crop = crop_region(image, region)
        except Exception as exc:
            logger.warning("Cannot crop region %s for OCR: %s", region.region_type, exc)
            continue

        if crop.width < 10 or crop.height < 10:
            continue

        words = ocr_provider.extract_text(crop)
        if not words:
            continue

        # Group words into lines (by y-coordinate proximity) → reassemble text
        full_text = _words_to_text(words)
        avg_confidence = sum(w["confidence"] for w in words) / len(words)

        entities.append(ExtractedEntity(
            entity_type=f"{region.region_type.value}_text",
            value=full_text,
            sheet_number=sheet_number,
            page_index=page_index,
            region_type=region.region_type,
            bbox=region.bbox,
            extraction_method=ExtractionMethod.OCR,
            confidence=round(avg_confidence, 3),
            raw_source=full_text,
        ))

    return entities


def _words_to_text(words: list[dict]) -> str:
    """Reassemble words into lines using y-coordinate clustering."""
    if not words:
        return ""
    sorted_words = sorted(words, key=lambda w: (w["bbox"]["y0"], w["bbox"]["x0"]))
    lines: list[list[str]] = []
    current_line: list[str] = []
    prev_y = sorted_words[0]["bbox"]["y0"]
    line_height_estimate = max(
        w["bbox"]["y1"] - w["bbox"]["y0"] for w in sorted_words[:10]
    ) if sorted_words else 20

    for word in sorted_words:
        y = word["bbox"]["y0"]
        if abs(y - prev_y) > line_height_estimate * 0.6:
            if current_line:
                lines.append(current_line)
            current_line = []
            prev_y = y
        current_line.append(word["text"])

    if current_line:
        lines.append(current_line)

    return "\n".join(" ".join(line) for line in lines)
