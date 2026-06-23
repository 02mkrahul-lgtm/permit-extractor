"""Evaluation harness: run the pipeline against a ground-truth directory.

Ground-truth directory format:
    ground_truth/
        sheet_001.pdf          (or sheet_001.png for a single raster page)
        sheet_001_gt.json      (ground truth annotations)
        sheet_002.pdf
        sheet_002_gt.json
        ...

Ground-truth JSON format (_gt.json):
    {
        "sheet_path": "sheet_001.pdf",
        "entities": [
            {"entity_type": "title_sheet_number", "expected_value": "A-101"},
            {"entity_type": "title_discipline",   "expected_value": "ARCHITECTURAL"},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from permit_extractor.config import PipelineConfig
from permit_extractor.evaluation.metrics import compute_metrics, format_metrics_report
from permit_extractor.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def run_eval(
    ground_truth_dir: str,
    config: PipelineConfig,
    output_path: Optional[str] = None,
    feedback_store=None,
) -> dict:
    """Run the pipeline on every PDF in ground_truth_dir and score against GT.

    Returns a metrics dict. Writes a Markdown report to output_path if provided.
    """
    gt_dir = Path(ground_truth_dir)
    gt_files = sorted(gt_dir.glob("*_gt.json"))

    if not gt_files:
        raise ValueError(f"No *_gt.json files found in {gt_dir}")

    all_predictions: list[dict] = []
    all_ground_truth: list[dict] = []
    skipped = 0

    for gt_file in gt_files:
        with open(gt_file) as fh:
            gt_data = json.load(fh)

        sheet_path_rel = gt_data.get("sheet_path", "")
        pdf_path = gt_dir / sheet_path_rel
        if not pdf_path.exists():
            logger.warning("PDF not found for GT file %s: %s", gt_file.name, pdf_path)
            skipped += 1
            continue

        try:
            result = run_pipeline(str(pdf_path), config, feedback_store=feedback_store)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", pdf_path, exc)
            skipped += 1
            continue

        predictions = [
            {"entity_type": e.entity_type, "value": e.value}
            for e in result.all_entities()
        ]
        all_predictions.extend(predictions)
        all_ground_truth.extend(gt_data.get("entities", []))

    metrics = compute_metrics(all_predictions, all_ground_truth)
    metrics["eval_meta"] = {
        "gt_files_processed": len(gt_files) - skipped,
        "gt_files_skipped": skipped,
        "total_predictions": len(all_predictions),
        "total_ground_truth": len(all_ground_truth),
    }

    if output_path:
        report = format_metrics_report(metrics)
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info("Evaluation report written to %s", output_path)

    return metrics
