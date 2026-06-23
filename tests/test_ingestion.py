"""Tests for ingestion stage."""

import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from permit_extractor.ingestion.sheet_detector import detect_and_render


def _make_mock_page(text: str = "", width: float = 612.0, height: float = 792.0):
    page = MagicMock()
    page.get_text.return_value = text
    page.rect = MagicMock()
    page.rect.width = width
    page.rect.height = height

    # Mock pixmap
    pix = MagicMock()
    pix.width = 2550
    pix.height = 3300
    # Create a small white image as samples
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    pix.samples = img.resize((2550, 3300)).tobytes()
    page.get_pixmap.return_value = pix
    return page


class TestSheetDetector:
    def test_detects_text_layer(self):
        page = _make_mock_page(text="A" * 200)
        info, img = detect_and_render(page, page_index=0, dpi=300, char_threshold=50)
        assert info.has_text_layer is True
        assert info.text_layer_coverage > 0

    def test_detects_raster(self):
        page = _make_mock_page(text="")
        info, img = detect_and_render(page, page_index=0, dpi=300, char_threshold=50)
        assert info.has_text_layer is False
        assert info.text_layer_coverage == 0.0

    def test_image_returned(self):
        page = _make_mock_page(text="hello world " * 20)
        info, img = detect_and_render(page, page_index=0)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_sheet_info_fields(self):
        page = _make_mock_page()
        info, _ = detect_and_render(page, page_index=2, dpi=150)
        assert info.page_index == 2
        assert info.render_dpi == 150
        assert info.width_pt == 612.0
