"""Inject eval.json run summary into markdown marker regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _split_hash(eval_data: dict[str, Any]) -> str:
    split = eval_data.get("split")
    if isinstance(split, dict):
        return str(split.get("hash", "—"))
    return "—"


def _current_state_body(eval_data: dict[str, Any]) -> str:
    run = eval_data.get("run", {})
    if not isinstance(run, dict):
        run = {}
    lat = eval_data.get("latency_exact", {})
    if not isinstance(lat, dict):
        lat = {}
    blocks = eval_data.get("active_blocks", {})
    if not isinstance(blocks, dict):
        blocks = {}
    lines = [
        "## Current state (from eval.json)",
        "",
        f"- **Generated (UTC):** {run.get('timestamp_utc', '—')}",
        f"- **Git SHA:** {run.get('git_sha', '—')}",
        f"- **Smoke run:** {eval_data.get('smoke', run.get('smoke', eval_data.get('fast_eval')))}",
        f"- **Split hash:** {_split_hash(eval_data)}",
        f"- **Queries:** tune {eval_data.get('tune_queries')}, "
        f"validation {eval_data.get('validation_queries')}, "
        f"test {eval_data.get('test_queries')}",
        f"- **Selected variant:** {eval_data.get('selected_variant', 'binary')}",
        f"- **Shipped transform:** {eval_data.get('shipped_transform', '—')}",
        f"- **Image coverage:** {eval_data.get('image_coverage')}",
        f"- **Image weight in tuning:** "
        f"{'zeroed (<5% coverage)' if eval_data.get('image_weight_zeroed') else 'active'}",
        f"- **Latency /find_similar_products (lean):** "
        f"p50 {lat.get('find_similar_p50_ms', lat.get('p50_ms', '—'))} ms, "
        f"p95 {lat.get('find_similar_p95_ms', lat.get('p95_ms', '—'))} ms",
        f"- **Latency full-catalog w/ metadata:** "
        f"p50 {lat.get('metadata_p50_ms', '—')} ms, "
        f"p95 {lat.get('metadata_p95_ms', '—')} ms",
        f"- **Active blocks (pair-active or coverage):** {blocks}",
        "",
    ]
    return "\n".join(lines)


def render_current_state_markers(eval_data: dict[str, Any], path: Path) -> None:
    start = "<!-- current_state:start -->"
    end = "<!-- current_state:end -->"
    text = path.read_text()
    body = f"{start}\n{_current_state_body(eval_data)}\n{end}"
    if start in text and end in text:
        before, _ = text.split(start, 1)
        _, after = text.split(end, 1)
        path.write_text(before + body + after)
    else:
        path.write_text(text.rstrip() + "\n\n" + body + "\n")
