#!/usr/bin/env python3
"""Construction Permit Set Extractor — CLI.

Commands:
    process     Extract structured data from a permit PDF
    eval        Run against a ground-truth directory and report accuracy
    feedback    Manage the feedback loop (apply, report, exemplars)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("permit_extractor")


@click.group()
def cli():
    """Construction Permit Set Extractor v1."""
    pass


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="./output", show_default=True,
              help="Directory for output files.")
@click.option("--dpi", default=300, show_default=True,
              help="Render DPI for sheet images.")
@click.option("--model", default=None, help="VLM model name (overrides config).")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "anthropic"]),
              help="VLM provider (overrides config).")
@click.option("--stages", default="all", show_default=True,
              help="Comma-separated stages to run: ingest,segment,vector,ocr,vlm,validate,report. "
                   "Or 'all'.")
@click.option("--no-vlm", is_flag=True, default=False,
              help="Skip VLM calls (vector+OCR extraction only).")
@click.option("--config", "config_path", default=None,
              help="Path to config.yaml (auto-detected if omitted).")
def process(pdf_path, output_dir, dpi, model, provider, stages, no_vlm, config_path):
    """Extract structured data from a permit PDF.

    Outputs:
      {stem}_extracted.json  — full structured extraction with provenance
      {stem}_summary.md      — human-readable summary for QA
      {stem}_review.json     — pre-populated review file (sorted by confidence)
      vlm_log.jsonl          — VLM call log (cost tracking)
    """
    from permit_extractor.config import load_config
    from permit_extractor.pipeline import run_pipeline
    from permit_extractor.reporting.json_writer import write_json
    from permit_extractor.reporting.markdown_report import write_markdown
    from permit_extractor.reporting.review_file import write_review_file

    cfg = load_config(
        config_path,
        dpi=dpi,
        output_dir=output_dir,
        vlm_model=model,
        vlm_provider=provider,
    )
    if no_vlm:
        cfg.run_vlm = False

    if stages != "all":
        stage_set = {s.strip() for s in stages.split(",")}
        cfg.run_segmentation = "segment" in stage_set
        cfg.run_vector = "vector" in stage_set
        cfg.run_ocr = "ocr" in stage_set
        cfg.run_vlm = "vlm" in stage_set and not no_vlm
        cfg.run_cross_check = "validate" in stage_set

    # Optionally load feedback store for exemplar injection
    feedback_store = None
    try:
        from permit_extractor.feedback.store import FeedbackStore
        feedback_store = FeedbackStore(cfg.feedback_db_path)
    except Exception:
        pass

    click.echo(f"Processing: {pdf_path}")
    click.echo(f"Output dir: {output_dir}")
    click.echo(f"VLM: {'disabled' if no_vlm else cfg.vlm_model + ' via ' + cfg.vlm_provider}")

    result = run_pipeline(pdf_path, cfg, feedback_store=feedback_store)

    stem = Path(pdf_path).stem
    json_path = write_json(result, output_dir, stem)
    md_path = write_markdown(result, output_dir, stem)
    review_path = write_review_file(result, output_dir, stem)

    m = result.metrics
    click.echo(f"\n✓ Done in {m.elapsed_seconds}s")
    click.echo(f"  Sheets: {m.total_pages} ({m.pages_with_text_layer} vector, {m.pages_raster} raster)")
    click.echo(f"  Entities: {len(result.all_entities())}")
    click.echo(f"  Needs review: {len(result.needs_review())}")
    click.echo(f"  VLM calls: {m.vlm_calls} | Est. cost: ${m.vlm_estimated_cost_usd:.4f}")
    click.echo(f"\n  → {json_path}")
    click.echo(f"  → {md_path}")
    click.echo(f"  → {review_path}")

    if feedback_store:
        feedback_store.close()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("ground_truth_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", default="eval_report.md", show_default=True,
              help="Path to write the evaluation Markdown report.")
@click.option("--model", default=None)
@click.option("--provider", default=None, type=click.Choice(["openai", "anthropic"]))
@click.option("--config", "config_path", default=None)
def eval(ground_truth_dir, output, model, provider, config_path):
    """Run pipeline on a ground-truth directory and report accuracy.

    Ground-truth directory must contain pairs of PDF + *_gt.json files.
    See permit_extractor/evaluation/harness.py for the GT JSON format.
    """
    from permit_extractor.config import load_config
    from permit_extractor.evaluation.harness import run_eval

    cfg = load_config(config_path, vlm_model=model, vlm_provider=provider)
    feedback_store = None
    try:
        from permit_extractor.feedback.store import FeedbackStore
        feedback_store = FeedbackStore(cfg.feedback_db_path)
    except Exception:
        pass

    click.echo(f"Evaluating against: {ground_truth_dir}")
    metrics = run_eval(ground_truth_dir, cfg, output_path=output, feedback_store=feedback_store)

    o = metrics["overall"]
    meta = metrics.get("eval_meta", {})
    click.echo(f"\n✓ Evaluation complete")
    click.echo(f"  Sheets processed: {meta.get('gt_files_processed', '?')}")
    click.echo(f"  Overall — P: {o['precision']:.3f}  R: {o['recall']:.3f}  F1: {o['f1']:.3f}")
    click.echo(f"  Report: {output}")

    if feedback_store:
        feedback_store.close()


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

@cli.group()
def feedback():
    """Manage the feedback loop: apply corrections, report improvements."""
    pass


@feedback.command("apply")
@click.argument("review_json", type=click.Path(exists=True))
@click.option("--db", default=None, help="Feedback SQLite DB path (overrides config).")
@click.option("--config", "config_path", default=None)
def feedback_apply(review_json, db, config_path):
    """Ingest a completed review file into the feedback store.

    Reads the _review.json file (filled in by a human reviewer), stores
    corrections in SQLite, grows the eval set, and creates exemplars for
    incorrect extractions.
    """
    from permit_extractor.config import load_config
    from permit_extractor.feedback.store import FeedbackStore
    from permit_extractor.feedback.pattern_catalog import PatternCatalog

    cfg = load_config(config_path)
    db_path = db or cfg.feedback_db_path
    store = FeedbackStore(db_path)

    with open(review_json) as fh:
        data = json.load(fh)

    items = data.get("items", [])
    reviewed = [i for i in items if i.get("status") != "pending"]
    click.echo(f"Ingesting {len(reviewed)} reviewed items from {review_json}")

    inserted = store.ingest_corrections(reviewed)

    # Mine new failure patterns from accumulated corrections
    catalog = PatternCatalog(store)
    new_patterns = catalog.mine_new_patterns()

    stats = store.correction_stats()
    click.echo(f"\n✓ Applied {inserted} corrections")
    click.echo(f"  Correct: {stats.get('correct', 0)}  "
               f"Incorrect: {stats.get('incorrect', 0)}  "
               f"Missing: {stats.get('missing', 0)}")
    if new_patterns:
        click.echo(f"  New failure patterns detected: {new_patterns}")

    store.close()


@feedback.command("report")
@click.option("--db", default=None)
@click.option("--config", "config_path", default=None)
def feedback_report(db, config_path):
    """Show accuracy trends and failure pattern status."""
    from permit_extractor.config import load_config
    from permit_extractor.feedback.store import FeedbackStore
    from permit_extractor.feedback.confidence_calibrator import (
        update_calibration_from_corrections, get_calibration_report
    )

    cfg = load_config(config_path)
    db_path = db or cfg.feedback_db_path

    if not Path(db_path).exists():
        click.echo("No feedback store found. Run `feedback apply` first.")
        return

    store = FeedbackStore(db_path)
    update_calibration_from_corrections(store)

    stats = store.correction_stats()
    total = stats.get("total", 0)
    correct = stats.get("correct", 0)
    accuracy = correct / total if total > 0 else 0.0

    click.echo("\n=== Feedback Report ===\n")
    click.echo(f"Total reviewed: {total}")
    click.echo(f"  Correct:   {correct}")
    click.echo(f"  Incorrect: {stats.get('incorrect', 0)}")
    click.echo(f"  Missing:   {stats.get('missing', 0)}")
    click.echo(f"  Accuracy:  {accuracy:.1%}\n")

    patterns = store.get_active_patterns()
    if patterns:
        click.echo("--- Active Failure Patterns ---\n")
        for p in patterns:
            click.echo(f"  [{p['occurrence_count']}x] {p['pattern_name']}: {p['description']}")
    else:
        click.echo("No failure patterns on record yet.")

    click.echo("\n--- Confidence Calibration ---\n")
    click.echo(get_calibration_report(store))

    gt_count = len(store.get_eval_ground_truth())
    click.echo(f"\nGround-truth eval set size: {gt_count} items")

    store.close()


@feedback.command("exemplars")
@click.argument("action", type=click.Choice(["list"]))
@click.option("--entity-type", default=None)
@click.option("--db", default=None)
@click.option("--config", "config_path", default=None)
def feedback_exemplars(action, entity_type, db, config_path):
    """List stored few-shot exemplars."""
    from permit_extractor.config import load_config
    from permit_extractor.feedback.store import FeedbackStore

    cfg = load_config(config_path)
    db_path = db or cfg.feedback_db_path

    if not Path(db_path).exists():
        click.echo("No feedback store found.")
        return

    store = FeedbackStore(db_path)
    rows = store._conn.execute("""
        SELECT entity_type, region_type, input_context, created_at FROM exemplars
        ORDER BY created_at DESC LIMIT 50
    """).fetchall()

    if not rows:
        click.echo("No exemplars stored yet.")
    else:
        for r in rows:
            if entity_type and r["entity_type"] != entity_type:
                continue
            click.echo(f"  [{r['entity_type']} / {r['region_type']}] created={r['created_at'][:10]}")

    store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
