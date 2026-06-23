"""Tests for core data models."""

import json
import pytest
from permit_extractor.models.entities import ExtractedEntity, ExtractionMethod
from permit_extractor.models.regions import BoundingBox
from permit_extractor.models.regions import Region, RegionType, SheetInfo
from permit_extractor.models.results import Check, ExtractionResult, RunMetrics, SheetResult
from permit_extractor.models.corrections import CorrectionRecord, CorrectionStatus


def make_bbox(page: int = 0) -> BoundingBox:
    return BoundingBox(x0=10.0, y0=10.0, x1=200.0, y1=100.0, page_index=page)


def make_entity(**kwargs) -> ExtractedEntity:
    defaults = dict(
        entity_type="title_sheet_number",
        value="A-101",
        sheet_number="A-101",
        page_index=0,
        region_type=RegionType.TITLE_BLOCK,
        bbox=make_bbox(),
        extraction_method=ExtractionMethod.VECTOR,
        confidence=0.95,
    )
    defaults.update(kwargs)
    return ExtractedEntity(**defaults)


class TestBoundingBox:
    def test_width_height(self):
        b = BoundingBox(0, 0, 100, 50, page_index=0)
        assert b.width() == 100
        assert b.height() == 50

    def test_roundtrip(self):
        b = BoundingBox(1.5, 2.5, 100.0, 200.0, page_index=3)
        assert BoundingBox.from_dict(b.to_dict()) == b


class TestExtractedEntity:
    def test_to_dict_keys(self):
        e = make_entity()
        d = e.to_dict()
        assert "entity_id" in d
        assert "provenance" not in d  # no extra nesting
        assert d["extraction_method"] == "vector"
        assert d["region_type"] == "title_block"

    def test_entity_id_unique(self):
        e1 = make_entity()
        e2 = make_entity()
        assert e1.entity_id != e2.entity_id

    def test_serialise_to_json(self):
        e = make_entity()
        json_str = json.dumps(e.to_dict(), default=str)
        loaded = json.loads(json_str)
        assert loaded["value"] == "A-101"


class TestExtractionResult:
    def _make_result(self) -> ExtractionResult:
        info = SheetInfo(0, True, 0.9, 612, 792, 300, 2550, 3300)
        sheet = SheetResult(0, info)
        sheet.entities = [make_entity()]
        sheet.sheet_number = "A-101"
        result = ExtractionResult(pdf_path="test.pdf")
        result.sheets.append(sheet)
        return result

    def test_all_entities(self):
        r = self._make_result()
        assert len(r.all_entities()) == 1

    def test_entities_by_type(self):
        r = self._make_result()
        assert len(r.entities_by_type("title_sheet_number")) == 1
        assert len(r.entities_by_type("nonexistent")) == 0

    def test_needs_review_low_confidence(self):
        r = self._make_result()
        r.sheets[0].entities[0].confidence = 0.5
        assert len(r.needs_review()) == 1

    def test_needs_review_high_confidence(self):
        r = self._make_result()
        assert len(r.needs_review()) == 0

    def test_to_dict_serialisable(self):
        r = self._make_result()
        json.dumps(r.to_dict(), default=str)


class TestCorrectionRecord:
    def test_roundtrip(self):
        r = CorrectionRecord(
            run_id="run1",
            entity_type="title_sheet_number",
            predicted_value="A-101",
            corrected_value="A-102",
            status=CorrectionStatus.INCORRECT,
        )
        d = r.to_dict()
        assert d["status"] == "incorrect"
        restored = CorrectionRecord.from_dict(d)
        assert restored.status == CorrectionStatus.INCORRECT
        assert restored.corrected_value == "A-102"
