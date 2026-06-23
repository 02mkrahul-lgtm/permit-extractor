"""Tests for extraction stages."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from permit_extractor.models.entities import ExtractedEntity, CrossCheckStatus, ExtractionMethod
from permit_extractor.models.regions import BoundingBox, Region, RegionType, SheetInfo
from permit_extractor.validation.cross_checker import cross_check_entities, _text_similarity


def make_bbox():
    return BoundingBox(0, 0, 100, 100, page_index=0)


def make_entity(method, entity_type="notes_text", value="hello world", confidence=0.9):
    return ExtractedEntity(
        entity_type=entity_type,
        value=value,
        sheet_number="A-101",
        page_index=0,
        region_type=RegionType.NOTES,
        bbox=make_bbox(),
        extraction_method=method,
        confidence=confidence,
    )


class TestCrossChecker:
    def test_matched(self):
        v = make_entity(ExtractionMethod.VECTOR, value="Type V-B construction")
        vlm = make_entity(ExtractionMethod.VLM, value="type v-b construction")
        entities, checks = cross_check_entities([v, vlm])
        assert v.cross_check_status == CrossCheckStatus.MATCHED
        assert len(checks) == 0

    def test_mismatch(self):
        v = make_entity(ExtractionMethod.VECTOR, value="Type V-B construction occupancy A-2")
        vlm = make_entity(ExtractionMethod.VLM, value="Completely different text here in the notes")
        entities, checks = cross_check_entities([v, vlm])
        assert v.cross_check_status == CrossCheckStatus.MISMATCH
        assert len(checks) == 1
        assert checks[0].check_type == "cross_check"

    def test_vlm_only(self):
        vlm = make_entity(ExtractionMethod.VLM)
        entities, checks = cross_check_entities([vlm])
        assert vlm.cross_check_status == CrossCheckStatus.VLM_ONLY

    def test_vector_only(self):
        v = make_entity(ExtractionMethod.VECTOR)
        entities, checks = cross_check_entities([v])
        assert v.cross_check_status == CrossCheckStatus.VECTOR_ONLY


class TestTextSimilarity:
    def test_identical(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        sim = _text_similarity("apple banana cherry", "dog cat fish")
        assert sim == 0.0

    def test_partial(self):
        sim = _text_similarity("hello world foo", "hello world bar")
        assert 0.3 < sim < 1.0

    def test_empty(self):
        assert _text_similarity("", "") == 1.0
        assert _text_similarity("hello", "") == 0.0


class TestOCRExtractor:
    def test_words_to_text(self):
        from permit_extractor.extraction.ocr_extractor import _words_to_text
        words = [
            {"text": "Hello", "confidence": 0.9, "bbox": {"x0": 10, "y0": 10, "x1": 50, "y1": 25}},
            {"text": "World", "confidence": 0.85, "bbox": {"x0": 55, "y0": 10, "x1": 100, "y1": 25}},
            {"text": "Foo", "confidence": 0.95, "bbox": {"x0": 10, "y0": 40, "x1": 40, "y1": 55}},
        ]
        text = _words_to_text(words)
        assert "Hello World" in text
        assert "Foo" in text
