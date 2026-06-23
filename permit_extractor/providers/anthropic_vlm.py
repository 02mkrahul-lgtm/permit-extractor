"""Anthropic Claude vision provider.

Swappable alternative to OpenAIVLMProvider. Select via config.yaml:
    vlm_provider: anthropic
    vlm_model: claude-sonnet-4-6
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Optional

import anthropic
from PIL import Image
from pydantic import BaseModel

from permit_extractor.providers.base import VLMProvider
from permit_extractor.providers.openai_vlm import _format_examples, _heuristic_confidence

logger = logging.getLogger(__name__)

_COST_PER_1K = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-opus-4-8": {"input": 0.015, "output": 0.075},
}


class AnthropicVLMProvider(VLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        log_path: Optional[str] = None,
        max_image_side: int = 2048,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._log_path = log_path
        self._max_image_side = max_image_side

    @property
    def model_name(self) -> str:
        return self._model

    def extract(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        few_shot_examples: Optional[list[dict]] = None,
    ) -> tuple[BaseModel, float]:
        image_b64 = _encode_image(image, max_side=self._max_image_side)

        # Build content
        content: list[dict] = []
        if few_shot_examples:
            content.append({"type": "text", "text": _format_examples(few_shot_examples)})
        content.append({"type": "text", "text": user_prompt})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            },
        })

        schema_json = output_schema.model_json_schema()
        json_instruction = (
            f"\n\nRespond with ONLY valid JSON conforming to this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n"
            f"Do not include any text outside the JSON object."
        )
        content.append({"type": "text", "text": json_instruction})

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        elapsed = time.monotonic() - t0

        raw_text = response.content[0].text if response.content else "{}"
        # Strip markdown fences if present
        raw_json = raw_text.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```", 2)[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        usage = response.usage
        cost = _estimate_cost(self._model, usage.input_tokens, usage.output_tokens)
        self._log_call(
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cost_usd=cost,
            elapsed=elapsed,
            raw_response=raw_json,
        )

        try:
            parsed = output_schema.model_validate_json(raw_json)
        except Exception as exc:
            logger.warning("Schema validation failed (%s); attempting lenient parse", exc)
            parsed = output_schema.model_validate(json.loads(raw_json))

        confidence = _heuristic_confidence(parsed, usage.output_tokens)
        return parsed, confidence

    def _log_call(self, **kwargs) -> None:
        if not self._log_path:
            return
        entry = {"timestamp": time.time(), **kwargs}
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def _encode_image(image: Image.Image, max_side: int = 2048) -> str:
    img = image.copy()
    if max(img.width, img.height) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, {"input": 0.003, "output": 0.015})
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
