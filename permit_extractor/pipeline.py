"""Pipeline orchestrator: runs all stages in sequence for a PDF.

Usage:
    from permit_extractor.pipeline import run_pipeline
    result = run_pipeline("my_permit.pdf", config)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from permit_extractor.config import PipelineConfig
from permit_extractor.ingestion.pdf_loader import iter_pages, page_count
from permit_extractor.ingestion.sheet_detector import detect_and_render
from permit_extractor.models.results import ExtractionResult, RunMetrics, SheetResult
from permit_extractor.segmentation.layout_segmenter import segment_sheet
from permit_extractor.validation.cross_checker import cross_check_entities

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_path: str,
    config: PipelineConfig,
    feedback_store=None,   # feedback.store.FeedbackStore | None
) -> ExtractionResult:
    """Process a PDF permit set end-to-end.

    Args:
        pdf_path: Path to the input PDF.
        config: Pipeline configuration.
        feedback_store: Optional live FeedbackStore for exemplar/pattern injection.

    Returns:
        ExtractionResult with all sheets, entities, checks, and metrics.
    """
    t_start = time.monotonic()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    log_path = str(out_dir / "vlm_log.jsonl") if config.log_vlm_calls else None

    # --- Build providers -----------------------------------------------
    vlm_provider = _build_vlm_provider(config, log_path)
    ocr_provider = _build_ocr_provider(config)

    # --- Build extractors ----------------------------------------------
    from permit_extractor.extraction.vlm_extractor import VLMExtractor
    crops_dir = str(out_dir / f"{stem}_crops") if config.save_region_crops else None
    exemplar_retriever = None
    pattern_catalog = None
    if feedback_store:
        from permit_extractor.feedback.exemplar_retriever import ExemplarRetriever
        from permit_extractor.feedback.pattern_catalog import PatternCatalog
        exemplar_retriever = ExemplarRetriever(feedback_store)
        pattern_catalog = PatternCatalog(feedback_store)

    vlm_extractor = VLMExtractor(
        provider=vlm_provider,
        exemplar_retriever=exemplar_retriever,
        pattern_catalog=pattern_catalog,
        crops_output_dir=crops_dir,
    ) if config.run_vlm else None

    # --- Initialise result object --------------------------------------
    result = ExtractionResult(
        pdf_path=str(pdf_path),
        config_snapshot=config.as_snapshot(),
    )
    metrics = result.metrics
    metrics.total_pages = page_count(pdf_path)
    metrics.model_used = config.vlm_model

    # --- Process pages one at a time (streaming) ----------------------
    for page_index, page in iter_pages(pdf_path):
        logger.info("Processing page %d/%d", page_index + 1, metrics.total_pages)

        # Stage 1: Detect text layer + render image
        sheet_info, image = detect_and_render(
            page, page_index, dpi=config.dpi,
            char_threshold=config.text_layer_char_threshold,
        )
        if sheet_info.has_text_layer:
            metrics.pages_with_text_layer += 1
        else:
            metrics.pages_raster += 1

        sheet_result = SheetResult(page_index=page_index, sheet_info=sheet_info)

        # Stage 2: Layout segmentation
        regions = []
        if config.run_segmentation:
            regions = segment_sheet(image, sheet_info)
        else:
            # Minimal fallback: one region covering the whole page
            from permit_extractor.segmentation.layout_segmenter import _segment_heuristic
            regions = _segment_heuristic(image, sheet_info)
        sheet_result.regions = regions

        entities = []

        # Stage 3a: Vector text extraction
        if config.run_vector and sheet_info.has_text_layer:
            from permit_extractor.extraction.vector_extractor import extract_text_entities
            vector_ents = extract_text_entities(page, regions, page_index)
            entities.extend(vector_ents)

        # Stage 3b: OCR fallback
        elif config.run_ocr and not sheet_info.has_text_layer and ocr_provider:
            from permit_extractor.extraction.ocr_extractor import extract_ocr_entities
            ocr_ents = extract_ocr_entities(image, regions, ocr_provider, page_index)
            entities.extend(ocr_ents)

        # Stage 4: VLM semantic extraction
        if config.run_vlm and vlm_extractor:
            vlm_ents = vlm_extractor.extract_regions(image, regions, page_index)
            # Update VLM metrics from the provider log (approximate from entity count)
            metrics.vlm_calls += len([r for r in regions
                                      if r.region_type.value in ("title_block", "schedule", "notes")])
            entities.extend(vlm_ents)

        # Stage 5: Cross-check
        if config.run_cross_check and entities:
            entities, page_checks = cross_check_entities(entities)
            result.checks.extend(page_checks)

        # Resolve sheet number from title entities
        sheet_number = _resolve_sheet_number(entities)
        sheet_result.sheet_number = sheet_number
        sheet_result.sheet_title = _resolve_field(entities, "title_sheet_title")
        sheet_result.discipline = _resolve_field(entities, "title_discipline")

        # Back-fill sheet_number on all entities
        for e in entities:
            if e.sheet_number == "UNKNOWN":
                e.sheet_number = sheet_number

        sheet_result.entities = entities
        result.sheets.append(sheet_result)

    metrics.elapsed_seconds = round(time.monotonic() - t_start, 2)
    return result


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def _build_vlm_provider(config: PipelineConfig, log_path: Optional[str]):
    if not config.run_vlm:
        return None
    if config.vlm_provider == "anthropic":
        from permit_extractor.providers.anthropic_vlm import AnthropicVLMProvider
        return AnthropicVLMProvider(
            api_key=config.anthropic_api_key,
            model=config.vlm_model,
            log_path=log_path,
        )
    from permit_extractor.providers.openai_vlm import OpenAIVLMProvider
    return OpenAIVLMProvider(
        api_key=config.openai_api_key,
        model=config.vlm_model,
        log_path=log_path,
    )


def _build_ocr_provider(config: PipelineConfig):
    if not config.run_ocr:
        return None
    try:
        from permit_extractor.providers.tesseract_ocr import TesseractOCRProvider
        return TesseractOCRProvider()
    except Exception as exc:
        logger.warning("Tesseract OCR provider unavailable (%s); OCR disabled", exc)
        return None


# ---------------------------------------------------------------------------
# Entity value resolution helpers
# ---------------------------------------------------------------------------

def _resolve_sheet_number(entities) -> str:
    for e in entities:
        if e.entity_type == "title_sheet_number" and e.value:
            return str(e.value).strip()
    return "UNKNOWN"


def _resolve_field(entities, entity_type: str) -> str:
    for e in entities:
        if e.entity_type == entity_type and e.value:
            return str(e.value).strip()
    return ""
