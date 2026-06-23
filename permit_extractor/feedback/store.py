"""SQLite-backed feedback store.

Five tables (see plan for schema):
  - corrections        human review decisions
  - exemplars          curated few-shot examples for prompt injection
  - failure_patterns   aggregated named failure patterns
  - eval_ground_truth  growing ground-truth eval set
  - confidence_calibration  predicted vs actual accuracy buckets
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class FeedbackStore:
    def __init__(self, db_path: str = "./permit_extractor_feedback.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS corrections (
            correction_id   TEXT PRIMARY KEY,
            run_id          TEXT,
            entity_id       TEXT,
            entity_type     TEXT,
            page_index      INTEGER,
            region_type     TEXT,
            sheet_number    TEXT,
            extraction_method TEXT,
            predicted_confidence REAL,
            predicted_value TEXT,   -- JSON
            corrected_value TEXT,   -- JSON
            status          TEXT,
            region_crop_path TEXT,
            raw_source      TEXT,
            reviewer_note   TEXT,
            reviewed_at     TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS exemplars (
            entry_id         TEXT PRIMARY KEY,
            entity_type      TEXT,
            region_type      TEXT,
            sheet_discipline TEXT,
            input_context    TEXT,
            corrected_output TEXT,  -- JSON
            created_at       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_exemplars_type
            ON exemplars(entity_type, region_type);

        CREATE TABLE IF NOT EXISTS failure_patterns (
            pattern_id          TEXT PRIMARY KEY,
            pattern_name        TEXT UNIQUE,
            entity_type         TEXT,
            region_type         TEXT,
            description         TEXT,
            prompt_injection    TEXT,
            occurrence_count    INTEGER DEFAULT 0,
            first_seen          TEXT,
            last_seen           TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_ground_truth (
            gt_id           TEXT PRIMARY KEY,
            entity_type     TEXT,
            sheet_path      TEXT,
            page_index      INTEGER,
            region_type     TEXT,
            expected_value  TEXT,   -- JSON
            source          TEXT,   -- 'human' | 'correction'
            correction_id   TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS confidence_calibration (
            bucket          TEXT PRIMARY KEY,   -- e.g. '0.5-0.6'
            predicted_low   REAL,
            predicted_high  REAL,
            correct_count   INTEGER DEFAULT 0,
            total_count     INTEGER DEFAULT 0,
            last_updated    TEXT
        );
        """)
        self._conn.commit()

    # --- Correction ingestion ------------------------------------------

    def ingest_corrections(self, records: list[dict]) -> int:
        """Insert or update correction records. Returns count inserted."""
        inserted = 0
        for r in records:
            if r.get("status") == "pending":
                continue  # skip unreviewed
            self._conn.execute("""
                INSERT OR REPLACE INTO corrections
                (correction_id, run_id, entity_id, entity_type, page_index,
                 region_type, sheet_number, extraction_method, predicted_confidence,
                 predicted_value, corrected_value, status, region_crop_path,
                 raw_source, reviewer_note, reviewed_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get("correction_id"), r.get("run_id"), r.get("entity_id"),
                r.get("entity_type"), r.get("page_index"), r.get("region_type"),
                r.get("sheet_number"), r.get("extraction_method"),
                r.get("predicted_confidence"),
                json.dumps(r.get("predicted_value")),
                json.dumps(r.get("corrected_value")),
                r.get("status"), r.get("region_crop_path"), r.get("raw_source"),
                r.get("reviewer_note"), r.get("reviewed_at"), r.get("created_at"),
            ))
            inserted += 1

            # Auto-grow eval set for confirmed corrections
            if r.get("status") in ("correct", "incorrect"):
                self._add_to_eval_set(r)

            # Store as exemplar for incorrect → corrected cases
            if r.get("status") == "incorrect" and r.get("corrected_value") is not None:
                self._store_exemplar(r)

        self._conn.commit()
        return inserted

    def _add_to_eval_set(self, r: dict) -> None:
        import uuid, datetime
        expected = r.get("corrected_value") if r.get("status") == "incorrect" else r.get("predicted_value")
        self._conn.execute("""
            INSERT OR IGNORE INTO eval_ground_truth
            (gt_id, entity_type, sheet_path, page_index, region_type,
             expected_value, source, correction_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), r.get("entity_type"), r.get("run_id"),
            r.get("page_index"), r.get("region_type"),
            json.dumps(expected), "correction", r.get("correction_id"),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ))

    def _store_exemplar(self, r: dict) -> None:
        import uuid, datetime
        self._conn.execute("""
            INSERT OR IGNORE INTO exemplars
            (entry_id, entity_type, region_type, sheet_discipline,
             input_context, corrected_output, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), r.get("entity_type"), r.get("region_type"), "",
            r.get("raw_source", ""), json.dumps(r.get("corrected_value")),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ))

    # --- Exemplar retrieval -------------------------------------------

    def get_exemplars(self, entity_type: str, region_type: str, limit: int = 3) -> list[dict]:
        rows = self._conn.execute("""
            SELECT input_context, corrected_output FROM exemplars
            WHERE entity_type = ? AND region_type = ?
            ORDER BY created_at DESC LIMIT ?
        """, (entity_type, region_type, limit)).fetchall()
        return [
            {
                "input_context": row["input_context"],
                "corrected_output": json.loads(row["corrected_output"]) if row["corrected_output"] else None,
            }
            for row in rows
        ]

    # --- Failure patterns ---------------------------------------------

    def get_active_patterns(self, region_type: Optional[str] = None) -> list[dict]:
        if region_type:
            rows = self._conn.execute("""
                SELECT * FROM failure_patterns WHERE region_type = ?
                ORDER BY occurrence_count DESC
            """, (region_type,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT * FROM failure_patterns ORDER BY occurrence_count DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def upsert_pattern(
        self,
        pattern_name: str,
        entity_type: str,
        region_type: str,
        description: str,
        prompt_injection: str,
    ) -> None:
        import uuid, datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute("""
            INSERT INTO failure_patterns
                (pattern_id, pattern_name, entity_type, region_type, description,
                 prompt_injection, occurrence_count, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(pattern_name) DO UPDATE SET
                occurrence_count = occurrence_count + 1,
                last_seen = excluded.last_seen,
                prompt_injection = excluded.prompt_injection
        """, (str(uuid.uuid4()), pattern_name, entity_type, region_type,
              description, prompt_injection, now, now))
        self._conn.commit()

    # --- Eval ground truth --------------------------------------------

    def get_eval_ground_truth(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM eval_ground_truth").fetchall()
        return [dict(r) for r in rows]

    # --- Confidence calibration ---------------------------------------

    def record_prediction_outcome(
        self, predicted_confidence: float, was_correct: bool
    ) -> None:
        import datetime
        bucket_low = round(int(predicted_confidence * 10) / 10, 1)
        bucket_high = round(bucket_low + 0.1, 1)
        bucket = f"{bucket_low:.1f}-{bucket_high:.1f}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute("""
            INSERT INTO confidence_calibration
                (bucket, predicted_low, predicted_high, correct_count, total_count, last_updated)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(bucket) DO UPDATE SET
                correct_count = correct_count + ?,
                total_count = total_count + 1,
                last_updated = excluded.last_updated
        """, (bucket, bucket_low, bucket_high, 1 if was_correct else 0, now,
              1 if was_correct else 0))
        self._conn.commit()

    def get_calibration_table(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT bucket, predicted_low, predicted_high,
                   correct_count, total_count,
                   CASE WHEN total_count > 0
                        THEN CAST(correct_count AS REAL) / total_count
                        ELSE NULL END AS observed_accuracy
            FROM confidence_calibration
            ORDER BY predicted_low
        """).fetchall()
        return [dict(r) for r in rows]

    # --- Stats --------------------------------------------------------

    def correction_stats(self) -> dict:
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status='correct' THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN status='incorrect' THEN 1 ELSE 0 END) as incorrect,
                SUM(CASE WHEN status='missing' THEN 1 ELSE 0 END) as missing
            FROM corrections WHERE status != 'pending'
        """).fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        self._conn.close()
