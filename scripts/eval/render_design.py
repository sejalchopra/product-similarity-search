"""Render docs/DESIGN.md eval region from eval.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt_ci(d: dict[str, object] | None) -> str:
    if not d:
        return "—"
    return f"{d.get('mean', 0):.3f} [{d.get('lo', 0):.3f}, {d.get('hi', 0):.3f}]"


def _ranking_case_table(case: dict[str, object]) -> str:
    if case.get("error"):
        return f"_Demo ranking case: {case.get('error')}_"
    qname = case.get("query_name") or case.get("query_id") or "—"
    lines = [
        f"Query: **{qname}** (`{case.get('query_id', '—')}`).",
        "",
        "| Rank | Before (binary) | After (selected) |",
        "|------|-----------------|------------------|",
    ]
    before = case.get("binary_top10", [])
    after = case.get("selected_top10", [])
    if not isinstance(before, list) or not isinstance(after, list):
        return "_No demo ranking table in eval.json._"
    for i in range(max(len(before), len(after))):
        b = before[i] if i < len(before) else {}
        a = after[i] if i < len(after) else {}
        if isinstance(b, dict) and isinstance(a, dict):
            bs = b.get("score")
            as_ = a.get("score")
            bsc = f"{float(bs):.3f}" if isinstance(bs, (int, float)) else "—"
            asc = f"{float(as_):.3f}" if isinstance(as_, (int, float)) else "—"
            lines.append(
                f"| {i + 1} | {b.get('name', '—')} ({bsc}) | "
                f"{a.get('name', '—')} ({asc}) |"
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
    cases = eval_data.get("cases") if isinstance(eval_data.get("cases"), dict) else {}
    demo_ranking = cases.get("demo_ranking", {}) if isinstance(cases, dict) else {}

    split = eval_data.get("split") if isinstance(eval_data.get("split"), dict) else {}
    smoke = bool(eval_data.get("smoke"))
    capped = bool(eval_data.get("capped"))
    if smoke:
        run_label = "smoke (capped)"
    elif capped:
        run_label = "capped (not smoke)"
    else:
        run_label = "full (uncapped)"
    cap_line = (
        f"tune **{eval_data.get('tune_queries')}** / {split.get('tune_queries_full')}, "
        f"validation **{eval_data.get('validation_queries')}** / {split.get('validation_queries_full')}, "
        f"test **{eval_data.get('test_queries')}** / {split.get('test_queries_full')}"
    )

    body = f"""{start}
## Problem and data (from eval.json)

- **Eval run:** {run_label}. Query caps: {cap_line}.
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

Validation scores for transforms and variants: `eval.json` → `validation_transform_ndcg10_pooled`, `validation_variant_ndcg10_pooled`.

Pair-active rates: color {pair.get("color")}, price {pair.get("price")}, rating {pair.get("rating")}, weight {pair.get("weight")}, image {pair.get("image")}.

Dedup merge: {eval_data.get("dedup_merge", {})}.

## Example ranking (binary vs selected, generated)

{_ranking_case_table(demo_ranking if isinstance(demo_ranking, dict) else {})}

{end}"""
    before, _ = text.split(start, 1)
    _, after = text.split(end, 1)
    design_path.write_text(before + body + after)
