"""VLM-based semantic extraction from region crops.

For each region:
  - Crop the region from the sheet image
  - Retrieve few-shot exemplars from the feedback store
  - Inject failure-pattern warnings from the pattern catalog
  - Send to VLM with the appropriate schema
  - Convert the structured response to ExtractedEntity objects
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from permit_extractor.models.entities import ExtractedEntity, ExtractionMethod
from permit_extractor.models.regions import BoundingBox
from permit_extractor.models.regions import Region, RegionType
from permit_extractor.providers.base import VLMProvider
from permit_extractor.schemas.title_block import TitleBlockSchema
from permit_extractor.schemas.schedule import ScheduleSchema
from permit_extractor.schemas.notes import NotesSchema
from permit_extractor.segmentation.layout_segmenter import crop_region

logger = logging.getLogger(__name__)

# Map region type → (schema, system_prompt_module, user_prompt_module)
_REGION_CONFIG: dict[RegionType, tuple] = {}

def _get_region_config():
    """Lazy import of prompts to avoid circular imports."""
    if not _REGION_CONFIG:
        from permit_extractor.prompts import title_block, schedule, notes
        _REGION_CONFIG[RegionType.TITLE_BLOCK] = (TitleBlockSchema, title_block.SYSTEM, title_block.USER)
        _REGION_CONFIG[RegionType.SCHEDULE] = (ScheduleSchema, schedule.SYSTEM, schedule.USER)
        _REGION_CONFIG[RegionType.NOTES] = (NotesSchema, notes.SYSTEM, notes.USER)
    return _REGION_CONFIG


class VLMExtractor:
    def __init__(
        self,
        provider: VLMProvider,
        exemplar_retriever=None,   # feedback.exemplar_retriever.ExemplarRetriever | None
        pattern_catalog=None,      # feedback.pattern_catalog.PatternCatalog | None
        crops_output_dir: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._exemplar_retriever = exemplar_retriever
        self._pattern_catalog = pattern_catalog
        self._crops_dir = Path(crops_output_dir) if crops_output_dir else None

    def extract_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        page_index: int,
        sheet_number: str = "UNKNOWN",
    ) -> list[ExtractedEntity]:
        """Run VLM extraction on all supported region types."""
        entities: list[ExtractedEntity] = []
        config = _get_region_config()

        for region in regions:
            if region.region_type not in config:
                continue  # DRAWING_BODY, LEGEND — skip VLM
            if region.region_type == RegionType.DRAWING_BODY:
                continue

            schema_cls, sys_prompt, user_prompt = config[region.region_type]

            # Get crop
            try:
                crop = crop_region(image, region)
            except Exception as exc:
                logger.warning("Could not crop region %s on page %d: %s",
                               region.region_type, page_index, exc)
                continue

            if crop.width < 10 or crop.height < 10:
                logger.debug("Skipping tiny crop for %s on page %d", region.region_type, page_index)
                continue

            # Save crop for feedback/review
            crop_path: Optional[str] = None
            if self._crops_dir:
                crop_path = self._save_crop(crop, page_index, region.region_type)

            # Retrieve exemplars
            exemplars: list[dict] = []
            if self._exemplar_retriever:
                try:
                    exemplars = self._exemplar_retriever.retrieve(
                        entity_type=region.region_type.value,
                        region_type=region.region_type.value,
                    )
                except Exception:
                    pass

            # Inject failure pattern warnings
            final_sys_prompt = sys_prompt
            if self._pattern_catalog:
                try:
                    warnings = self._pattern_catalog.get_warnings(region.region_type.value)
                    if warnings:
                        final_sys_prompt = sys_prompt + "\n\n" + warnings
                except Exception:
                    pass

            # Call VLM (with retry)
            try:
                parsed, confidence = self._call_vlm_with_retry(
                    crop, final_sys_prompt, user_prompt, schema_cls, exemplars
                )
            except Exception as exc:
                logger.error("VLM extraction failed for %s on page %d: %s",
                             region.region_type, page_index, exc)
                continue

            # Convert schema output to entities
            new_entities = _schema_to_entities(
                parsed=parsed,
                confidence=confidence,
                region=region,
                page_index=page_index,
                sheet_number=sheet_number,
                crop_path=crop_path,
            )
            entities.extend(new_entities)

        return entities

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_vlm_with_retry(self, crop, sys_prompt, user_prompt, schema_cls, exemplars):
        return self._provider.extract(
            image=crop,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            output_schema=schema_cls,
            few_shot_examples=exemplars if exemplars else None,
        )

    def _save_crop(self, crop: Image.Image, page_index: int, region_type: RegionType) -> str:
        self._crops_dir.mkdir(parents=True, exist_ok=True)
        fname = f"page{page_index:03d}_{region_type.value}.jpg"
        path = self._crops_dir / fname
        crop.save(str(path), format="JPEG", quality=90)
        return str(path)


# ---------------------------------------------------------------------------
# Convert VLM schema output → ExtractedEntity list
# ---------------------------------------------------------------------------

def _schema_to_entities(
    parsed,
    confidence: float,
    region: Region,
    page_index: int,
    sheet_number: str,
    crop_path: Optional[str],
) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    data = parsed.model_dump()

    if isinstance(parsed, TitleBlockSchema):
        # Each field in title block → one entity
        for field_name, value in data.items():
            if value is None:
                continue
            entities.append(ExtractedEntity(
                entity_type=f"title_{field_name}",
                value=value,
                sheet_number=sheet_number,
                page_index=page_index,
                region_type=region.region_type,
                bbox=region.bbox,
                extraction_method=ExtractionMethod.VLM,
                confidence=confidence,
                raw_source=crop_path,
            ))

    elif isinstance(parsed, ScheduleSchema):
        if data.get("rows"):
            entities.append(ExtractedEntity(
                entity_type="schedule_table",
                value={
                    "table_type": data.get("table_type"),
                    "title": data.get("title"),
                    "headers": data.get("headers", []),
                    "rows": parsed.to_row_dicts(),
                    "notes": data.get("notes"),
                },
                sheet_number=sheet_number,
                page_index=page_index,
                region_type=region.region_type,
                bbox=region.bbox,
                extraction_method=ExtractionMethod.VLM,
                confidence=confidence,
                raw_source=crop_path,
            ))

    elif isinstance(parsed, NotesSchema):
        # One entity for the whole notes block, plus individual clause entities
        entities.append(ExtractedEntity(
            entity_type="general_notes",
            value=data,
            sheet_number=sheet_number,
            page_index=page_index,
            region_type=region.region_type,
            bbox=region.bbox,
            extraction_method=ExtractionMethod.VLM,
            confidence=confidence,
            raw_source=crop_path,
        ))
        for clause in data.get("clauses", []):
            if clause.get("text"):
                entities.append(ExtractedEntity(
                    entity_type="note_clause",
                    value=clause,
                    sheet_number=sheet_number,
                    page_index=page_index,
                    region_type=region.region_type,
                    bbox=region.bbox,
                    extraction_method=ExtractionMethod.VLM,
                    confidence=confidence,
                ))

    return entities
