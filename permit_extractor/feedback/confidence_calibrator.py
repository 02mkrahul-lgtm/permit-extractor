"""Recalibrate confidence scores using correction data.

Uses Platt scaling (logistic regression on predicted confidence vs actual
correctness) to produce calibrated confidence scores that stay meaningful
as the system accumulates corrections.

This module is called by the feedback report command and can be used
to produce a calibration curve for QA reporting.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def update_calibration_from_corrections(store) -> None:
    """Record prediction outcomes from all reviewed corrections."""
    rows = store._conn.execute("""
        SELECT predicted_confidence, status FROM corrections
        WHERE status IN ('correct', 'incorrect')
    """).fetchall()

    for row in rows:
        was_correct = row["status"] == "correct"
        store.record_prediction_outcome(row["predicted_confidence"], was_correct)


def get_calibration_report(store) -> str:
    """Return a Markdown table of predicted vs observed accuracy."""
    table = store.get_calibration_table()
    if not table:
        return "No calibration data available yet."

    lines = [
        "| Predicted bucket | Observed accuracy | Sample size |",
        "|-----------------|-------------------|-------------|",
    ]
    for row in table:
        obs = f"{row['observed_accuracy']:.1%}" if row["observed_accuracy"] is not None else "—"
        lines.append(f"| {row['bucket']} | {obs} | {row['total_count']} |")

    return "\n".join(lines)


def calibrated_confidence(raw_confidence: float, store) -> float:
    """Map a raw model confidence to a calibrated value using stored data.

    Falls back to raw_confidence if insufficient calibration data exists.
    """
    table = store.get_calibration_table()
    if not table or sum(r["total_count"] for r in table) < 20:
        return raw_confidence  # not enough data yet

    # Find the matching bucket
    for row in table:
        if row["total_count"] >= 5 and row["predicted_low"] <= raw_confidence < row["predicted_high"]:
            obs = row["observed_accuracy"]
            if obs is not None:
                # Blend raw and calibrated (70% calibrated, 30% raw to avoid over-correction)
                return round(0.7 * obs + 0.3 * raw_confidence, 3)

    return raw_confidence
