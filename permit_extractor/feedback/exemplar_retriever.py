"""Retrieve few-shot exemplars from the feedback store for prompt injection."""

from __future__ import annotations

from permit_extractor.feedback.store import FeedbackStore


class ExemplarRetriever:
    def __init__(self, store: FeedbackStore, default_limit: int = 3) -> None:
        self._store = store
        self._limit = default_limit

    def retrieve(self, entity_type: str, region_type: str) -> list[dict]:
        """Return up to `limit` exemplars for this (entity_type, region_type) pair.

        Returns a list of dicts with keys:
            input_context:    text or crop path context
            corrected_output: the correct structured output (dict)
        """
        return self._store.get_exemplars(
            entity_type=entity_type,
            region_type=region_type,
            limit=self._limit,
        )
