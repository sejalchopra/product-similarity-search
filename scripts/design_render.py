"""Render docs/DESIGN.md eval region from eval.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt_ci(d: dict[str, object] | None) -> str:
    if not d:
        return "—"
    return f"{d.get('mean', 0):.3f} [{d.get('lo', 0):.3f}, {d.get('hi', 0):.3f}]"


def _blazze_table(case: dict[str, object]) -> str:
    if case.get("error"):
        return f"_Blazze case: {case.get('error')}_"
    lines = [
        "| Rank | Before (binary) | After (selected) |",
        "|------|-----------------|------------------|",
    ]
    before = case.get("binary_top10", [])
    after = case.get("selected_top10", [])
    if not isinstance(before, list) or not isinstance(after, list):
        return "_No Blazze table in eval.json._"
    for i in range(max(len(before), len(after))):
        b = before[i] if i < len(before) else {}
        a = after[i] if i < len(after) else {}
        if isinstance(b, dict) and isinstance(a, dict):
            lines.append(
                f"| {i + 1} | {b.get('name', '—')} ({b.get('score', '—')}) | "
                f"{a.get('name', '—')} ({a.get('score', '—')}) |"
            )
    changed = case.get("ranking_changed")
    lines.append("")
    lines.append(f"Ranking changed: **{changed}**.")
    return "\n".join(lines)


def render_design(eval_data: dict[str, Any], design_path: Path) -> None:
    text = design_path.read_text()
    start = "<!-- eval:start -->"
    end = "<!-- eval:end -->"
    if start not in text or end not in text:
        raise RuntimeError("DESIGN.md missing eval markers")

    shipped = str(eval_data.get("shipped_transform", "tail_log"))
    test = eval_data.get("test_results", {})
    fusion = test.get("fusion", {}) if isinstance(test, dict) else {}
    baselines = test.get("baselines", {}) if isinstance(test, dict) else {}
    mini = baselines.get("minilm_only", {}) if isinstance(baselines, dict) else {}
    pop = baselines.get("popularity", {}) if isinstance(baselines, dict) else {}
    pair = eval_data.get("pair_active_rates", {})
    tg = eval_data.get("title_group_hist", {})
    ws = eval_data.get("weight_search", {})
    unchanged = ws.get("unchanged_reason") if isinstance(ws, dict) else None
    blazze = (
        eval_data.get("cases", {}).get("blazze", {})
        if isinstance(eval_data.get("cases"), dict)
        else {}
    )

    body = f"""{start}
## Problem and data (from eval.json)

- **Catalog:** {eval_data.get("row_count")} rows.
- **Image coverage:** {eval_data.get("image_coverage")}.
- **Title groups:** {tg.get("unique_groups") if isinstance(tg, dict) else "—"} unique keys for variant dedup.
- **Splits:** tune {eval_data.get("tune_queries")}, validation {eval_data.get("validation_queries")}, test {eval_data.get("test_queries")} queries.

## Scoring

Shipped transform: **{shipped}** (chosen on validation). Selected variant: **{eval_data.get("selected_variant", "binary")}** — {eval_data.get("selection_reason", "")}.

Weight search: {unchanged or "search moved weights (see eval.json → weight_search.search_trace)."}

Image block in tuning: {"zeroed (coverage < 5%)" if eval_data.get("image_weight_zeroed") else "active"}.

## Test evaluation (co-purchase @ title_group)

Metrics below are computed **once** on the held-out test split after selection. Cluster-bootstrap 95% CIs.

| Method | NDCG@10 | Hit@10 |
|--------|---------|--------|
| Fusion (selected) | {_fmt_ci(fusion.get("ndcg10") if isinstance(fusion, dict) else None)} | {_fmt_ci(fusion.get("hit10") if isinstance(fusion, dict) else None)} |
| MiniLM-only | {_fmt_ci(mini.get("ndcg10") if isinstance(mini, dict) else None)} | {_fmt_ci(mini.get("hit10") if isinstance(mini, dict) else None)} |
| Popularity (same child) | {_fmt_ci(pop.get("ndcg10") if isinstance(pop, dict) else None)} | {_fmt_ci(pop.get("hit10") if isinstance(pop, dict) else None)} |

Validation scores for transforms and variants: `eval.json` → `validation_transform_ndcg10`, `validation_variant`.

Pair-active rates: color {pair.get("color")}, price {pair.get("price")}, rating {pair.get("rating")}, weight {pair.get("weight")}, image {pair.get("image")}.

Dedup merge: {eval_data.get("dedup_merge", {})}.

## Blazze before / after (generated)

{_blazze_table(blazze if isinstance(blazze, dict) else {})}

{end}"""
    before, _ = text.split(start, 1)
    _, after = text.split(end, 1)
    design_path.write_text(before + body + after)
