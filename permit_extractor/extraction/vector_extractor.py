"""Extract text from a PDF page's vector text layer with precise coordinates.

Uses PyMuPDF's get_text("dict") for per-word bounding boxes, and pdfplumber
for structured table detection in schedule regions.

All entities are tagged extraction_method=VECTOR.
"""

from __future__ import annotations

import logging
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from permit_extractor.models.entities import ExtractedEntity, ExtractionMethod
from permit_extractor.models.regions import BoundingBox
from permit_extractor.models.regions import Region, RegionType

logger = logging.getLogger(__name__)


def extract_text_entities(
    page: fitz.Page,
    regions: list[Region],
    page_index: int,
    sheet_number: str = "UNKNOWN",
) -> list[ExtractedEntity]:
    """Extract vector text from all regions on the page.

    For schedule regions, attempt pdfplumber table extraction.
    For other regions, extract text blocks via PyMuPDF.
    """
    entities: list[ExtractedEntity] = []

    for region in regions:
        if region.region_type == RegionType.DRAWING_BODY:
            # Don't send the full drawing body to VLM — skip detailed vector extraction
            continue

        # Clip the page to the region's PDF-point bbox
        clip = fitz.Rect(region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1)

        if region.region_type == RegionType.SCHEDULE:
            table_entities = _extract_schedule_table(page, region, clip, page_index, sheet_number)
            if table_entities:
                entities.extend(table_entities)
                continue  # prefer table extraction over raw text

        # Raw text blocks for non-schedule regions
        text_entities = _extract_text_blocks(page, region, clip, page_index, sheet_number)
        entities.extend(text_entities)

    return entities


def _extract_text_blocks(
    page: fitz.Page,
    region: Region,
    clip: fitz.Rect,
    page_index: int,
    sheet_number: str,
) -> list[ExtractedEntity]:
    """Extract text spans from a region as individual text entities."""
    entities: list[ExtractedEntity] = []

    try:
        raw = page.get_text("dict", clip=clip)
    except Exception as exc:
        logger.warning("PyMuPDF get_text failed for region %s on page %d: %s",
                       region.region_type, page_index, exc)
        return []

    full_text_lines: list[str] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            line_text = " ".join(span["text"] for span in line.get("spans", []))
            if line_text.strip():
                full_text_lines.append(line_text.strip())

    if not full_text_lines:
        return []

    full_text = "\n".join(full_text_lines)
    entities.append(ExtractedEntity(
        entity_type=f"{region.region_type.value}_text",
        value=full_text,
        sheet_number=sheet_number,
        page_index=page_index,
        region_type=region.region_type,
        bbox=region.bbox,
        extraction_method=ExtractionMethod.VECTOR,
        confidence=0.95,   # vector text is high fidelity
        raw_source=full_text,
    ))
    return entities


def _extract_schedule_table(
    page: fitz.Page,
    region: Region,
    clip: fitz.Rect,
    page_index: int,
    sheet_number: str,
) -> list[ExtractedEntity]:
    """Use pdfplumber to extract a structured table from a schedule region.

    Returns an empty list if pdfplumber finds no table (caller falls back
    to raw text extraction).
    """
    entities: list[ExtractedEntity] = []
    try:
        # pdfplumber uses a separate file handle — open from the same page's parent
        # We pass the PDF path via the page's parent document.
        doc: fitz.Document = page.parent
        pdf_bytes = doc.tobytes()

        with pdfplumber.open(__import__("io").BytesIO(pdf_bytes)) as plumber_pdf:
            plumber_page = plumber_pdf.pages[page_index]
            # Crop to the region
            cropped = plumber_page.crop(
                (clip.x0, clip.y0, clip.x1, clip.y1), relative=False
            )
            tables = cropped.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )
            if not tables:
                # Try text-based strategy as fallback
                tables = cropped.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "lines",
                    }
                )

            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = [str(h or "").strip() for h in table[0]]
                rows = []
                for row in table[1:]:
                    cells = {headers[i]: str(cell or "").strip()
                             for i, cell in enumerate(row) if i < len(headers)}
                    if any(v for v in cells.values()):
                        rows.append(cells)

                if rows:
                    entities.append(ExtractedEntity(
                        entity_type="schedule_table",
                        value={"headers": headers, "rows": rows},
                        sheet_number=sheet_number,
                        page_index=page_index,
                        region_type=RegionType.SCHEDULE,
                        bbox=region.bbox,
                        extraction_method=ExtractionMethod.VECTOR,
                        confidence=0.90,
                        raw_source=str(tables),
                    ))
    except Exception as exc:
        logger.debug("pdfplumber table extraction failed: %s", exc)

    return entities
