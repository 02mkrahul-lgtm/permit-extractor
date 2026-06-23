"""Evaluation metrics: per-entity-type accuracy, precision, recall."""

from __future__ import annotations

import re
from typing import Any


def compute_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Compare predicted entities against ground truth.

    Args:
        predictions: list of {entity_type, value} dicts from the pipeline
        ground_truth: list of {entity_type, expected_value} dicts from GT files

    Returns:
        {
            "overall": {"accuracy": float, "precision": float, "recall": float},
            "by_type": {entity_type: {"tp": int, "fp": int, "fn": int, ...}},
        }
    """
    by_type: dict[str, dict] = {}

    # Group ground truth by entity_type
    gt_by_type: dict[str, list] = {}
    for item in ground_truth:
        et = item["entity_type"]
        gt_by_type.setdefault(et, []).append(item["expected_value"])

    # Group predictions by entity_type
    pred_by_type: dict[str, list] = {}
    for item in predictions:
        et = item["entity_type"]
        pred_by_type.setdefault(et, []).append(item["value"])

    all_types = set(gt_by_type) | set(pred_by_type)
    total_tp = total_fp = total_fn = 0

    for et in sorted(all_types):
        gt_vals = gt_by_type.get(et, [])
        pred_vals = pred_by_type.get(et, [])

        tp = _count_matches(pred_vals, gt_vals)
        fp = len(pred_vals) - tp
        fn = len(gt_vals) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        by_type[et] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall
                  / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0.0)

    return {
        "overall": {
            "precision": round(overall_precision, 3),
            "recall": round(overall_recall, 3),
            "f1": round(overall_f1, 3),
        },
        "by_type": by_type,
    }


def _count_matches(preds: list[Any], gts: list[Any]) -> int:
    """Greedy matching: count how many predictions match a GT value."""
    remaining_gts = list(gts)
    matches = 0
    for pred in preds:
        for i, gt in enumerate(remaining_gts):
            if _values_match(pred, gt):
                matches += 1
                remaining_gts.pop(i)
                break
    return matches


def _values_match(pred: Any, gt: Any) -> bool:
    """Fuzzy value comparison."""
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False

    # Numeric: within 5% tolerance
    try:
        p_num = float(str(pred).replace(",", "").strip())
        g_num = float(str(gt).replace(",", "").strip())
        if g_num != 0:
            return abs(p_num - g_num) / abs(g_num) <= 0.05
        return p_num == g_num
    except (ValueError, TypeError):
        pass

    # String: case-insensitive, whitespace-normalised
    p_str = _normalise_str(str(pred))
    g_str = _normalise_str(str(gt))
    if p_str == g_str:
        return True

    # Partial match: one is a substring of the other (for long text fields)
    if len(p_str) > 5 and len(g_str) > 5:
        shorter, longer = (p_str, g_str) if len(p_str) <= len(g_str) else (g_str, p_str)
        if shorter in longer:
            return True

    return False


def _normalise_str(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_metrics_report(metrics: dict, run_id: str = "") -> str:
    """Return a Markdown table of metrics."""
    lines = [
        f"# Evaluation Metrics{' — Run ' + run_id if run_id else ''}",
        "",
        "## Overall",
        "",
        "| Precision | Recall | F1 |",
        "|-----------|--------|-----|",
    ]
    o = metrics["overall"]
    lines.append(f"| {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} |")
    lines += ["", "## Per Entity Type", ""]
    lines.append("| Entity Type | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|-------------|----|----|----|-----------| -------|-----|")
    for et, m in sorted(metrics["by_type"].items()):
        lines.append(
            f"| {et} | {m['tp']} | {m['fp']} | {m['fn']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"
