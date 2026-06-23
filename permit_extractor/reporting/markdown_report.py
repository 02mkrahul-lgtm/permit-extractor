"""Generate a human-readable extraction summary in Markdown.

Goal: a reviewer can read this and immediately know where the extraction is
weak, what was found, and what needs manual checking — without opening JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from permit_extractor.models.entities import CrossCheckStatus, ExtractionMethod
from permit_extractor.models.results import ExtractionResult


def write_markdown(result: ExtractionResult, output_dir: str, stem: str) -> Path:
    out = Path(output_dir) / f"{stem}_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(result), encoding="utf-8")
    return out


def _render(result: ExtractionResult) -> str:
    lines: list[str] = []
    m = result.metrics

    lines += [
        f"# Extraction Summary — {Path(result.pdf_path).name}",
        f"",
        f"Run ID: `{result.run_id}`  |  Timestamp: {result.timestamp}",
        f"Model: `{m.model_used}`  |  Elapsed: {m.elapsed_seconds}s",
        f"VLM calls: {m.vlm_calls}  |  Estimated cost: ${m.vlm_estimated_cost_usd:.4f}",
        f"",
        f"---",
        f"",
        f"## Sheet Inventory ({m.total_pages} pages)",
        f"",
        f"| Page | Sheet # | Title | Discipline | Layer | Regions |",
        f"|------|---------|-------|------------|-------|---------|",
    ]

    for s in result.sheets:
        layer = "vector" if s.sheet_info.has_text_layer else "raster/OCR"
        region_types = ", ".join(sorted({r.region_type.value for r in s.regions}))
        lines.append(
            f"| {s.page_index + 1} | {s.sheet_number} | {s.sheet_title or '—'} "
            f"| {s.discipline or '—'} | {layer} | {region_types} |"
        )

    lines += ["", "---", "", "## Entity Counts by Type", ""]

    # Tally by entity_type
    counts: dict[str, int] = {}
    for e in result.all_entities():
        counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
    for et, cnt in sorted(counts.items()):
        lines.append(f"- **{et}**: {cnt}")

    # Extraction method breakdown
    method_counts: dict[str, int] = {}
    for e in result.all_entities():
        method_counts[e.extraction_method.value] = method_counts.get(e.extraction_method.value, 0) + 1
    lines += ["", "### By Extraction Method", ""]
    for method, cnt in sorted(method_counts.items()):
        lines.append(f"- `{method}`: {cnt} entities")

    # Cross-check mismatches
    mismatches = [e for e in result.all_entities()
                  if e.cross_check_status == CrossCheckStatus.MISMATCH]
    lines += ["", "---", "", f"## Cross-Check Results", ""]
    if mismatches:
        lines.append(f"**{len(mismatches)} mismatch(es)** between vector text and VLM extraction:\n")
        for e in mismatches:
            delta = e.cross_check_delta or {}
            lines.append(
                f"- **Page {e.page_index + 1}** | `{e.entity_type}` | sheet `{e.sheet_number}`  "
                f"  similarity={delta.get('similarity', '?')}"
            )
            lines.append(f"  - Vector: `{str(delta.get('vector_value', ''))[:120]}`")
            lines.append(f"  - VLM:    `{str(delta.get('vlm_value', ''))[:120]}`")
    else:
        lines.append("No vector/VLM mismatches detected.")

    # Needs review list
    needs_review = result.needs_review()
    lines += ["", "---", "", f"## Needs Review ({len(needs_review)} items)", ""]
    if needs_review:
        lines.append("*Items with confidence < 0.7 or cross-check mismatch. Review these first.*\n")
        for e in sorted(needs_review, key=lambda x: x.confidence):
            flag = "MISMATCH" if e.cross_check_status == CrossCheckStatus.MISMATCH else f"conf={e.confidence:.2f}"
            lines.append(
                f"- [{flag}] **Page {e.page_index + 1}** | `{e.entity_type}` | "
                f"sheet `{e.sheet_number}` | method=`{e.extraction_method.value}`"
            )
            val_str = str(e.value)
            if len(val_str) > 120:
                val_str = val_str[:120] + "…"
            lines.append(f"  Value: `{val_str}`")
    else:
        lines.append("No items flagged for review.")

    # Checks summary
    lines += ["", "---", "", f"## Validation Checks ({len(result.checks)})", ""]
    if result.checks:
        for c in result.checks:
            lines.append(f"- [{c.severity.upper()}] {c.description}")
    else:
        lines.append("No checks generated.")

    # Run metrics
    lines += [
        "", "---", "", "## Run Metrics", "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total pages | {m.total_pages} |",
        f"| Pages with text layer | {m.pages_with_text_layer} |",
        f"| Raster/OCR pages | {m.pages_raster} |",
        f"| VLM calls | {m.vlm_calls} |",
        f"| Est. cost (USD) | ${m.vlm_estimated_cost_usd:.4f} |",
        f"| Elapsed | {m.elapsed_seconds}s |",
    ]

    return "\n".join(lines) + "\n"
