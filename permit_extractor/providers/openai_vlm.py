"""OpenAI vision provider (default).

Sends a base64-encoded region crop to GPT-4o-mini (or any configured model)
and forces JSON output conforming to the given Pydantic schema.

Logs every call to vlm_log.jsonl for cost tracking and debugging.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from permit_extractor.providers.base import VLMProvider

logger = logging.getLogger(__name__)

# Cost estimates per 1K tokens (as of mid-2025; update as pricing changes)
_COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
}


class OpenAIVLMProvider(VLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        log_path: Optional[str] = None,
        max_image_side: int = 2048,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
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
        # Encode image
        image_b64 = _encode_image(image, max_side=self._max_image_side)

        # Build user content
        content: list[dict] = []
        if few_shot_examples:
            example_text = _format_examples(few_shot_examples)
            content.append({"type": "text", "text": example_text})
        content.append({"type": "text", "text": user_prompt})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        schema_json = _make_openai_strict_schema(output_schema.model_json_schema())

        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": schema_json,
                    "strict": True,
                },
            },
            temperature=0,
            max_tokens=16384,
        )
        elapsed = time.monotonic() - t0

        raw_json = response.choices[0].message.content or "{}"
        usage = response.usage

        cost = _estimate_cost(self._model, usage.prompt_tokens, usage.completion_tokens)
        self._log_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            elapsed=elapsed,
            raw_response=raw_json,
        )

        try:
            parsed = output_schema.model_validate_json(raw_json)
        except Exception as exc:
            logger.warning("Schema validation failed (%s); attempting lenient parse", exc)
            try:
                parsed = output_schema.model_validate(json.loads(raw_json))
            except json.JSONDecodeError:
                # Response was truncated — return an empty schema instance with low confidence
                logger.warning("VLM response was truncated (hit max_tokens?); returning empty result")
                parsed = output_schema.model_construct()
                confidence = 0.1
                self._log_call(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    model=self._model, prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cost_usd=_estimate_cost(self._model, usage.prompt_tokens, usage.completion_tokens),
                    elapsed=elapsed, raw_response=raw_json, truncated=True,
                )
                return parsed, confidence

        confidence = _heuristic_confidence(parsed, usage.completion_tokens)
        return parsed, confidence

    def _log_call(self, **kwargs) -> None:
        if not self._log_path:
            return
        entry = {"timestamp": time.time(), **kwargs}
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def _encode_image(image: Image.Image, max_side: int = 2048) -> str:
    """Resize if needed then base64-encode as JPEG."""
    img = image.copy()
    if max(img.width, img.height) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _format_examples(examples: list[dict]) -> str:
    lines = ["Here are examples of correct extractions from similar regions:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"[Example {i}]")
        if "input_context" in ex:
            lines.append(f"Context: {ex['input_context']}")
        if "corrected_output" in ex:
            lines.append(f"Correct output: {json.dumps(ex['corrected_output'], indent=2)}")
        lines.append("")
    return "\n".join(lines)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, {"input": 0.001, "output": 0.003})
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


def _make_openai_strict_schema(schema: dict) -> dict:
    """Transform a Pydantic JSON schema to satisfy OpenAI strict mode requirements.

    OpenAI strict mode requires every object to have:
      - "additionalProperties": false
      - all properties listed in "required"
    Pydantic does not emit these by default.
    """
    import copy
    schema = copy.deepcopy(schema)
    _patch_object(schema)
    for def_schema in schema.get("$defs", {}).values():
        _patch_object(def_schema)
    return schema


def _patch_object(node: dict) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        if "properties" in node:
            # Force ALL properties into required — OpenAI strict mode demands this
            # even for fields with defaults. Optional values use null union types.
            node["required"] = list(node["properties"].keys())
        for prop in node.get("properties", {}).values():
            _patch_object(prop)
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(key, []):
            _patch_object(sub)
    if isinstance(node.get("items"), dict):
        _patch_object(node["items"])


def _heuristic_confidence(parsed: BaseModel, completion_tokens: int) -> float:
    """Estimate confidence from how many fields are populated."""
    data = parsed.model_dump()
    non_null = sum(1 for v in data.values() if v is not None and v != [] and v != {})
    total = len(data)
    if total == 0:
        return 0.5
    fill_ratio = non_null / total
    # Penalise very short responses (likely refusals or empty)
    token_penalty = 0.0 if completion_tokens > 20 else 0.3
    return round(max(0.1, fill_ratio - token_penalty), 3)
