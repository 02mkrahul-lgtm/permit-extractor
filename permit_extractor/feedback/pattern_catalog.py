"""Aggregate corrections into named failure patterns and inject them as prompts.

The catalog is queried before each VLM call so persistent failure patterns
are included as "Watch for X" warnings in the system prompt.
"""

from __future__ import annotations

import json
from collections import Counter

from permit_extractor.feedback.store import FeedbackStore


# Known failure patterns — seeded with common construction drawing issues.
# Additional patterns are mined from corrections automatically.
_SEED_PATTERNS: list[dict] = [
    {
        "pattern_name": "drops_last_schedule_row",
        "entity_type": "schedule_table",
        "region_type": "schedule",
        "description": "VLM drops the final row(s) of long schedules when the table extends to the crop edge.",
        "prompt_injection": "Watch for: Do not drop the last row(s) of the schedule even if they are at the very bottom edge of the image.",
    },
    {
        "pattern_name": "misreads_revision_triangle",
        "entity_type": "title_revision",
        "region_type": "title_block",
        "description": "VLM misreads revision triangles — confuses the triangle symbol with a letter.",
        "prompt_injection": "Watch for: Revision numbers inside triangles (△1, △2, etc.) — read the number inside the triangle, not the triangle symbol.",
    },
    {
        "pattern_name": "merged_header_columns",
        "entity_type": "schedule_table",
        "region_type": "schedule",
        "description": "Merged/spanned header cells cause VLM to produce wrong column count.",
        "prompt_injection": "Watch for: Some column headers span multiple sub-columns. List each sub-column as a separate header.",
    },
]


class PatternCatalog:
    def __init__(self, store: FeedbackStore) -> None:
        self._store = store
        self._seed_done = False

    def _ensure_seeded(self) -> None:
        if self._seed_done:
            return
        for p in _SEED_PATTERNS:
            # Only insert if not already present (upsert on name is fine)
            existing = self._store.get_active_patterns(p["region_type"])
            names = {x["pattern_name"] for x in existing}
            if p["pattern_name"] not in names:
                self._store.upsert_pattern(
                    pattern_name=p["pattern_name"],
                    entity_type=p["entity_type"],
                    region_type=p["region_type"],
                    description=p["description"],
                    prompt_injection=p["prompt_injection"],
                )
        self._seed_done = True

    def get_warnings(self, region_type: str) -> str:
        """Return a formatted block of "Watch for:" warnings for this region type.

        Returns empty string if no patterns apply.
        """
        self._ensure_seeded()
        patterns = self._store.get_active_patterns(region_type)
        if not patterns:
            return ""
        lines = ["Failure patterns observed on similar drawings — pay special attention:"]
        for p in patterns:
            lines.append(f"- {p['prompt_injection']}")
        return "\n".join(lines)

    def mine_new_patterns(self) -> int:
        """Analyse recent corrections and promote recurring errors to named patterns.

        Returns the number of new patterns created.
        """
        # Get all incorrect corrections grouped by entity_type
        rows = self._store._conn.execute("""
            SELECT entity_type, region_type, predicted_value, corrected_value
            FROM corrections WHERE status = 'incorrect'
        """).fetchall()

        if len(rows) < 3:
            return 0

        # Simple frequency-based mining: if the same entity_type fails ≥ 3 times,
        # create a pattern for it
        type_counter: Counter = Counter()
        for r in rows:
            type_counter[(r["entity_type"], r["region_type"])] += 1

        created = 0
        for (entity_type, region_type), count in type_counter.items():
            if count >= 3:
                name = f"recurring_error_{entity_type}"
                description = f"Entity type '{entity_type}' has failed {count} times in region '{region_type}'."
                injection = f"Watch for: '{entity_type}' extraction has been error-prone — double-check values carefully."
                self._store.upsert_pattern(
                    pattern_name=name,
                    entity_type=entity_type,
                    region_type=region_type,
                    description=description,
                    prompt_injection=injection,
                )
                created += 1

        return created
