"""Pipeline configuration.

Loaded from config.yaml at startup; can be overridden via CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # Rendering
    dpi: int = 300

    # VLM provider
    vlm_provider: str = "openai"          # "openai" | "anthropic"
    vlm_model: str = "gpt-4o-mini"

    # OCR provider
    ocr_provider: str = "tesseract"

    # Stage flags (set False to skip a stage)
    run_segmentation: bool = True
    run_vector: bool = True
    run_ocr: bool = True
    run_vlm: bool = True
    run_cross_check: bool = True

    # Text-layer detection threshold
    # A page is considered raster if get_text() yields fewer than this many chars
    text_layer_char_threshold: int = 50

    # Output
    output_dir: str = "./output"
    save_region_crops: bool = True  # save cropped region images alongside output

    # Feedback store
    feedback_db_path: str = "./permit_extractor_feedback.db"

    # Logging
    log_vlm_calls: bool = True

    # API keys (read from env if not set here)
    openai_api_key: Optional[str] = field(
        default=None,
        metadata={"env": "OPENAI_API_KEY"},
    )
    anthropic_api_key: Optional[str] = field(
        default=None,
        metadata={"env": "ANTHROPIC_API_KEY"},
    )

    def __post_init__(self) -> None:
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def as_snapshot(self) -> dict:
        """Serialisable config for embedding in ExtractionResult."""
        return {
            "dpi": self.dpi,
            "vlm_provider": self.vlm_provider,
            "vlm_model": self.vlm_model,
            "ocr_provider": self.ocr_provider,
            "run_segmentation": self.run_segmentation,
            "run_vector": self.run_vector,
            "run_ocr": self.run_ocr,
            "run_vlm": self.run_vlm,
            "run_cross_check": self.run_cross_check,
        }


def load_config(config_path: Optional[str] = None, **overrides) -> PipelineConfig:
    """Load config from YAML file then apply keyword overrides."""
    import yaml  # optional; fall back gracefully

    cfg: dict = {}
    search = [
        config_path,
        os.environ.get("PERMIT_EXTRACTOR_CONFIG"),
        "config.yaml",
        str(Path(__file__).parent.parent / "config.yaml"),
    ]
    for path in search:
        if path and Path(path).exists():
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            break

    cfg.update({k: v for k, v in overrides.items() if v is not None})
    # Remove keys not in PipelineConfig to avoid TypeError
    valid = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    cfg = {k: v for k, v in cfg.items() if k in valid}
    return PipelineConfig(**cfg)
