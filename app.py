"""Flask web server for permit-extractor.

Upload a PDF → pipeline runs → summary and entities displayed in browser.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import markdown2
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./web_output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store (fine for local dev)
jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    job_id = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    file.save(str(pdf_path))

    # Config from form
    run_vlm = request.form.get("run_vlm") == "on"
    dpi = int(request.form.get("dpi", 200))
    model = request.form.get("model", "gpt-4o-mini")
    api_key = request.form.get("api_key", "").strip() or os.environ.get("OPENAI_API_KEY", "")

    jobs[job_id] = {"status": "running", "started": time.time(), "filename": file.filename}

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, str(pdf_path), run_vlm, dpi, model, api_key),
        daemon=True,
    )
    thread.start()

    return redirect(url_for("job_status", job_id=job_id))


@app.route("/job/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return "Job not found", 404
    return render_template("job.html", job_id=job_id, job=job)


@app.route("/api/job/<job_id>/status")
def api_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


@app.route("/results/<job_id>")
def results(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return redirect(url_for("job_status", job_id=job_id))

    out_dir = OUTPUT_DIR / job_id
    stem = job.get("stem", "output")

    # Load summary markdown
    md_path = out_dir / f"{stem}_summary.md"
    summary_html = ""
    if md_path.exists():
        summary_html = markdown2.markdown(
            md_path.read_text(),
            extras=["tables", "fenced-code-blocks"],
        )

    # Load extracted entities
    json_path = out_dir / f"{stem}_extracted.json"
    entities = []
    metrics = {}
    sheets = []
    if json_path.exists():
        data = json.loads(json_path.read_text())
        metrics = data.get("metrics", {})
        for sheet in data.get("sheets", []):
            sheets.append({
                "number": sheet.get("sheet_number", "?"),
                "title": sheet.get("sheet_title", ""),
                "discipline": sheet.get("discipline", ""),
                "layer": "vector" if sheet.get("sheet_info", {}).get("has_text_layer") else "raster",
                "regions": ", ".join({r["region_type"] for r in sheet.get("regions", [])}),
                "entity_count": len(sheet.get("entities", [])),
            })
            for e in sheet.get("entities", []):
                val = e.get("value", "")
                if isinstance(val, dict):
                    val_display = json.dumps(val)[:200]
                elif isinstance(val, list):
                    val_display = json.dumps(val)[:200]
                else:
                    val_display = str(val)[:200]
                entities.append({
                    "sheet": sheet.get("sheet_number", "?"),
                    "page": e.get("page_index", 0) + 1,
                    "type": e.get("entity_type", ""),
                    "method": e.get("extraction_method", ""),
                    "confidence": f"{e.get('confidence', 0):.0%}",
                    "cross_check": e.get("cross_check_status") or "",
                    "value": val_display,
                })

    return render_template(
        "results.html",
        job_id=job_id,
        job=job,
        summary_html=summary_html,
        entities=entities,
        sheets=sheets,
        metrics=metrics,
        stem=stem,
    )


@app.route("/download/<job_id>/<filetype>")
def download(job_id, filetype):
    job = jobs.get(job_id)
    if not job:
        return "Not found", 404
    out_dir = OUTPUT_DIR / job_id
    stem = job.get("stem", "output")
    files = {
        "json": out_dir / f"{stem}_extracted.json",
        "review": out_dir / f"{stem}_review.json",
        "summary": out_dir / f"{stem}_summary.md",
    }
    path = files.get(filetype)
    if not path or not path.exists():
        return "File not found", 404
    return send_file(str(path), as_attachment=True)


def _run_pipeline_job(job_id, pdf_path, run_vlm, dpi, model, api_key):
    try:
        from permit_extractor.config import PipelineConfig
        from permit_extractor.pipeline import run_pipeline
        from permit_extractor.reporting.json_writer import write_json
        from permit_extractor.reporting.markdown_report import write_markdown
        from permit_extractor.reporting.review_file import write_review_file

        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf_path).stem

        cfg = PipelineConfig(
            dpi=dpi,
            vlm_provider="openai",
            vlm_model=model,
            run_vlm=run_vlm,
            output_dir=str(out_dir),
            save_region_crops=False,
            openai_api_key=api_key or None,
        )

        result = run_pipeline(pdf_path, cfg)

        write_json(result, str(out_dir), stem)
        write_markdown(result, str(out_dir), stem)
        write_review_file(result, str(out_dir), stem)

        m = result.metrics
        jobs[job_id].update({
            "status": "done",
            "stem": stem,
            "elapsed": m.elapsed_seconds,
            "pages": m.total_pages,
            "entities": len(result.all_entities()),
            "needs_review": len(result.needs_review()),
            "vlm_calls": m.vlm_calls,
        })

    except Exception as exc:
        import traceback
        jobs[job_id].update({
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
