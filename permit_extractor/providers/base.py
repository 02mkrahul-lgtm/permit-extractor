"""Abstract base classes for swappable VLM and OCR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image
from pydantic import BaseModel


class VLMProvider(ABC):
    """Interface for vision-language model providers.

    Implementations: OpenAIVLMProvider (openai_vlm.py), AnthropicVLMProvider (anthropic_vlm.py).
    Selected via config.yaml `vlm_provider`.
    """

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        few_shot_examples: Optional[list[dict]] = None,
    ) -> tuple[BaseModel, float]:
        """Send a region crop to the VLM and return structured output + confidence.

        Args:
            image: PIL Image of the cropped region.
            system_prompt: Instructions for the model.
            user_prompt: The specific extraction request, including any exemplars.
            output_schema: Pydantic model the JSON must conform to.
            few_shot_examples: Optional list of {"input": ..., "output": ...} dicts
                               to prepend as examples in the user message.

        Returns:
            (parsed_result, confidence)  where confidence is 0.0–1.0.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class OCRProvider(ABC):
    """Interface for OCR providers.

    Implementations: TesseractOCRProvider (tesseract_ocr.py).
    Selected via config.yaml `ocr_provider`.
    """

    @abstractmethod
    def extract_text(
        self,
        image: Image.Image,
    ) -> list[dict]:
        """Run OCR on a PIL Image and return a list of word-level results.

        Each dict in the return list has:
            text: str          — the recognised word
            confidence: float  — 0.0–1.0
            bbox: dict         — {x0, y0, x1, y1} in pixels
        """
        ...
