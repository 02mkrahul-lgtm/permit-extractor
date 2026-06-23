"""Tests for the feedback store and metrics."""

import json
import os
import tempfile
import pytest

from permit_extractor.feedback.store import FeedbackStore
from permit_extractor.evaluation.metrics import compute_metrics, _values_match


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_feedback.db")
    s = FeedbackStore(db)
    yield s
    s.close()


class TestFeedbackStore:
    def test_ingest_correct(self, store):
        records = [{
            "correction_id": "c1",
            "run_id": "run1",
            "entity_id": "e1",
            "entity_type": "title_sheet_number",
            "page_index": 0,
            "region_type": "title_block",
            "sheet_number": "A-101",
            "extraction_method": "vlm",
            "predicted_confidence": 0.9,
            "predicted_value": "A-101",
            "corrected_value": None,
            "status": "correct",
            "region_crop_path": None,
            "raw_source": None,
            "reviewer_note": None,
            "reviewed_at": None,
            "created_at": "2025-01-01T00:00:00Z",
        }]
        inserted = store.ingest_corrections(records)
        assert inserted == 1
        stats = store.correction_stats()
        assert stats["correct"] == 1
        assert stats["incorrect"] == 0

    def test_ingest_pending_skipped(self, store):
        records = [{"status": "pending", "correction_id": "c2"}]
        inserted = store.ingest_corrections(records)
        assert inserted == 0

    def test_ingest_incorrect_creates_exemplar(self, store):
        records = [{
            "correction_id": "c3",
            "run_id": "run1",
            "entity_id": "e3",
            "entity_type": "schedule_table",
            "page_index": 1,
            "region_type": "schedule",
            "sheet_number": "A-201",
            "extraction_method": "vlm",
            "predicted_confidence": 0.6,
            "predicted_value": {"rows": []},
            "corrected_value": {"rows": [{"Mark": "101", "Width": "3'-0\""}]},
            "status": "incorrect",
            "region_crop_path": None,
            "raw_source": "some context",
            "reviewer_note": "missed all rows",
            "reviewed_at": "2025-01-02T00:00:00Z",
            "created_at": "2025-01-01T00:00:00Z",
        }]
        store.ingest_corrections(records)
        exemplars = store.get_exemplars("schedule_table", "schedule")
        assert len(exemplars) == 1

    def test_confidence_calibration(self, store):
        store.record_prediction_outcome(0.8, True)
        store.record_prediction_outcome(0.8, False)
        table = store.get_calibration_table()
        assert len(table) == 1
        row = table[0]
        assert row["total_count"] == 2
        assert row["correct_count"] == 1
        assert abs(row["observed_accuracy"] - 0.5) < 0.01

    def test_pattern_upsert(self, store):
        store.upsert_pattern(
            "test_pattern", "schedule_table", "schedule",
            "A test pattern", "Watch for: test"
        )
        patterns = store.get_active_patterns("schedule")
        assert any(p["pattern_name"] == "test_pattern" for p in patterns)


class TestEvalMetrics:
    def test_exact_match(self):
        preds = [{"entity_type": "title_sheet_number", "value": "A-101"}]
        gts = [{"entity_type": "title_sheet_number", "expected_value": "A-101"}]
        m = compute_metrics(preds, gts)
        assert m["overall"]["precision"] == 1.0
        assert m["overall"]["recall"] == 1.0

    def test_no_match(self):
        preds = [{"entity_type": "title_sheet_number", "value": "X-999"}]
        gts = [{"entity_type": "title_sheet_number", "expected_value": "A-101"}]
        m = compute_metrics(preds, gts)
        assert m["overall"]["precision"] == 0.0
        assert m["overall"]["recall"] == 0.0

    def test_fuzzy_case_insensitive(self):
        assert _values_match("Type V-B", "type v-b")

    def test_fuzzy_numeric(self):
        assert _values_match(100.0, 100)
        assert _values_match("100", 100.0)
        assert not _values_match(100.0, 200.0)

    def test_partial_precision_recall(self):
        preds = [
            {"entity_type": "t", "value": "A"},
            {"entity_type": "t", "value": "B"},
            {"entity_type": "t", "value": "C"},  # false positive
        ]
        gts = [
            {"entity_type": "t", "expected_value": "A"},
            {"entity_type": "t", "expected_value": "B"},
            {"entity_type": "t", "expected_value": "D"},  # false negative
        ]
        m = compute_metrics(preds, gts)
        t = m["by_type"]["t"]
        assert t["tp"] == 2
        assert t["fp"] == 1
        assert t["fn"] == 1
