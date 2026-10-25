#!/usr/bin/env python3
"""Write docs/eval.json — single source of truth for DESIGN.md metrics."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.design_render import render_design
from scripts.eval_bootstrap import N_BOOT_DEFAULT, cluster_bootstrap_ci, paired_bootstrap_ci
from scripts.eval_checkpoint import CACHE_DIR, config_hash, load_stage, save_stage
from scripts.eval_fidelity import audit_pool_fidelity
from scripts.eval_guardrails import guardrail_query_arrays, guardrails_pass, mean_guardrails
from scripts.eval_pooled import (
    PooledQueryStore,
    profile_key,
    top10_groups_exact_on_pool,
    top10_groups_on_pool,
)
from scripts.eval_progress import StageTimer, print_runtime_table
from scripts.eval_progress import log as progress_log
from scripts.eval_progress import start as progress_start
from scripts.eval_reports import (
    blazze_comparison_table,
    top_title_groups,
)
from scripts.eval_selection import select_transform_on_validation, select_variant_on_validation
from scripts.eval_tune import (
    hit_at_k,
    mean_metrics_on_rows,
    ndcg_at_k,
    tune_weights_hybrid,
    weights_unchanged_reason,
)
from scripts.state_render import render_current_state_markers
from service.block_batch import compute_block_batch
from service.catalog import Catalog, load_catalog
from service.cleaning import dedup_merge_stats, normalize_title_for_group
from service.config import Settings
from service.explore import explore_similar
from service.scoring import BLOCK_NAMES, DEFAULT_WEIGHTS, FusionWeights
from service.scoring_profile import ScoringProfile
from service.similarity import find_similar_products
from service.transforms import TextImageTransform

_POOL_COUNTED: set[str] = set()

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "docs" / "DESIGN.md"
SUMMARY_PATH = ROOT / "docs" / "IMPLEMENTATION_SUMMARY.md"
OUT = ROOT / "docs" / "eval.json"
OUT_SMOKE = ROOT / "docs" / "eval.smoke.json"
TUNED_PATH = ROOT / "artifacts" / "tuned_weights.json"
TUNED_SMOKE_PATH = ROOT / "artifacts" / "tuned_weights.smoke.json"
LDJSON = (
    ROOT / "data" / "marketing_sample_for_amazon_com-amazon_fashion_products__"
    "20200201_20200430__30k_data.ldjson"
)
MIN_TUNE_QUERIES = 1500
IMAGE_COVERAGE_FLOOR = 0.05


def _git_sha() -> dict[str, str]:
    if not (ROOT / ".git").is_dir():
        return {
            "sha": "no-git",
            "status": "not a git repository (no .git directory; git init was not run)",
        }
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {"sha": out.strip(), "status": "ok"}
    except FileNotFoundError:
        return {"sha": "no-git", "status": "git executable not found"}
    except subprocess.CalledProcessError as e:
        return {"sha": "no-git", "status": f"git failed: {e.returncode}"}


def _split_caps(smoke: bool) -> tuple[int | None, int | None, int | None]:
    if smoke or os.environ.get("EVAL_SMOKE", "").strip() in ("1", "true", "yes"):
        return (
            int(os.environ.get("EVAL_MAX_TUNE", "400")),
            int(os.environ.get("EVAL_MAX_VAL", "200")),
            int(os.environ.get("EVAL_MAX_TEST", "200")),
        )
    legacy = os.environ.get("EVAL_MAX_QUERIES", "").strip()
    if legacy.isdigit():
        q = int(legacy)
        return q, q, q
    return None, None, None


def _groups_hash(train_g: set[int], val_g: set[int], test_g: set[int]) -> str:
    blob = json.dumps(
        {
            "train": sorted(train_g),
            "val": sorted(val_g),
            "test": sorted(test_g),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_or_build_store(
    catalog: Catalog,
    rows: list[int],
    relevants: dict[int, set[int]],
    transform: TextImageTransform,
    profile: ScoringProfile,
    beta: float,
    inject: bool,
    base_hash: str,
    stage_times: dict[str, float],
    *,
    record: bool = True,
) -> PooledQueryStore:
    stage = f"pool_{'train' if inject else 'eval'}_{transform.value}_{profile_key(profile)}_{beta}"
    h = config_hash(
        {
            "base": base_hash,
            "stage": stage,
            "rows": rows,
            "inject": inject,
            "image_sha256": catalog.manifest.get("image_sha256", ""),
        }
    )
    cached = load_stage(stage, h)
    bucket = "pool_build_train" if inject else "pool_build_eval"
    elapsed_path = CACHE_DIR / f"{stage}_{h}.sec"
    if isinstance(cached, PooledQueryStore):
        recorded = float(elapsed_path.read_text()) if elapsed_path.exists() else 0.0
        stage_times[stage] = recorded
        if record and stage not in _POOL_COUNTED:
            _POOL_COUNTED.add(stage)
            stage_times[bucket] = stage_times.get(bucket, 0.0) + recorded
            stage_times["pool_build"] = stage_times.get("pool_build", 0.0) + recorded
        progress_log(f"cache hit {stage} recorded_build={recorded:.1f}s")
        return cached
    timer = StageTimer(stage)
    store = PooledQueryStore.build(
        catalog,
        rows,
        relevants,
        transform,
        profile,
        beta,
        inject_relevant_groups=inject,
        timer=timer,
    )
    stage_times[stage] = timer.elapsed
    if record and stage not in _POOL_COUNTED:
        _POOL_COUNTED.add(stage)
        stage_times[bucket] = stage_times.get(bucket, 0.0) + timer.elapsed
        stage_times["pool_build"] = stage_times.get("pool_build", 0.0) + timer.elapsed
    stage_times["pool_build_this_run"] = stage_times.get("pool_build_this_run", 0.0) + timer.elapsed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    elapsed_path.write_text(str(timer.elapsed))
    save_stage(stage, h, store)
    return store


def _pooled_val_ndcg(
    store: PooledQueryStore,
    catalog: Catalog,
    rows: list[int],
    relevants: dict[int, set[int]],
    fw: FusionWeights,
) -> tuple[float, np.ndarray]:
    ndcgs: list[float] = []
    arr = np.full(len(rows), np.nan, dtype=np.float64)
    row_index = {r: i for i, r in enumerate(rows)}
    for qrow in rows:
        rel = relevants.get(qrow, set())
        if not rel:
            continue
        ent = store.by_row[qrow]
        groups = top10_groups_on_pool(catalog, qrow, ent.pool, ent.blocks, fw)
        n = ndcg_at_k(groups, rel, 10)
        arr[row_index[qrow]] = n
        ndcgs.append(n)
    return float(np.mean(ndcgs)) if ndcgs else 0.0, arr


def _pool_fidelity(
    catalog: Catalog,
    store: PooledQueryStore,
    rows: list[int],
    fw: FusionWeights,
    profile: ScoringProfile,
) -> dict[str, float]:
    match = 0
    total = 0
    for qrow in rows:
        ent = store.by_row[qrow]
        a = top10_groups_on_pool(catalog, qrow, ent.pool, ent.blocks, fw)
        b = top10_groups_exact_on_pool(catalog, qrow, ent.pool, fw, profile)
        if a == b:
            match += 1
        total += 1
    top10_eq_full = 0
    for qrow in rows:
        ent = store.by_row[qrow]
        pool_groups = top10_groups_on_pool(catalog, qrow, ent.pool, ent.blocks, fw)
        full_groups = _top10_groups_full(catalog, qrow, Settings(), fw, profile)
        if pool_groups == full_groups:
            top10_eq_full += 1
    n = max(total, 1)
    return {
        "matvec_equals_exact_on_pool": match / n,
        "pool_top10_equals_full_catalog_top10": top10_eq_full / max(len(rows), 1),
    }


def _relevant_coverage(store: PooledQueryStore, rows: list[int]) -> float:
    vals = [
        ent.relevant_in_pool
        for qrow in rows
        for ent in [store.by_row[qrow]]
        if ent.relevant_in_pool is not None
    ]
    return float(np.mean(vals)) if vals else 0.0


def _write_tuned_weights(
    path: Path,
    fw: FusionWeights,
    variant: str,
    source: str,
    split_info: dict[str, Any],
    *,
    selection_reason: str,
    transform_reason: str,
    guardrails: dict[str, Any],
) -> None:
    norm = fw.normalized()
    w_sum = float(sum(norm.w[b] for b in norm.w))
    payload = {
        "source": source,
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_sha(),
        "split": split_info,
        "selected_variant": variant,
        "text_image_transform": fw.text_image_transform.value,
        "beta": fw.beta,
        "w": {k: float(norm.w[k]) for k in norm.w},
        "w_sum": w_sum,
        "selection_reason": selection_reason,
        "transform_selection_reason": transform_reason,
        "guardrails": guardrails,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_microbenchmark(
    catalog: Catalog,
    tune_rows: list[int],
    relevants: dict[int, set[int]],
    stage_times: dict[str, float],
    base_hash: str,
) -> dict[str, float]:
    sample = tune_rows[: min(200, len(tune_rows))]
    if len(sample) < 5:
        return {}
    progress_log(f"benchmark: building pooled cache for {len(sample)} queries")
    t0 = time.perf_counter()
    store = _load_or_build_store(
        catalog,
        sample,
        relevants,
        TextImageTransform.TAIL_LOG,
        ScoringProfile(),
        1.0,
        True,
        base_hash,
        stage_times,
        record=False,
    )
    build_sec = time.perf_counter() - t0
    from scripts.eval_tune import TuneRunner

    runner = TuneRunner(catalog, store, sample, relevants, TextImageTransform.TAIL_LOG, 1.0, True)
    t1 = time.perf_counter()
    for _ in range(40):
        runner.eval_weights(dict(DEFAULT_WEIGHTS))
    weight_sec = (time.perf_counter() - t1) / 40.0
    gids = np.array([catalog.title_group[r] for r in sample], dtype=np.int64)
    fake = np.random.default_rng(0).random(len(sample))
    t2 = time.perf_counter()
    cluster_bootstrap_ci(fake, gids, n_boot=N_BOOT_DEFAULT)
    boot_sec = time.perf_counter() - t2
    t3 = time.perf_counter()
    _load_or_build_store(
        catalog,
        sample,
        relevants,
        TextImageTransform.TAIL_LOG,
        ScoringProfile(category_mode="affinity"),
        1.0,
        True,
        base_hash,
        stage_times,
    )
    aff_sec = time.perf_counter() - t3
    return {
        "pool_build_200_queries_sec": build_sec,
        "affinity_pool_build_200_queries_sec": aff_sec,
        "mean_weight_eval_sec": weight_sec,
        "bootstrap_2000_resamples_sec": boot_sec,
    }


def _norm_name(name: str) -> str:
    s = re.sub(r"\s+", " ", name.strip().lower())
    return s


def build_copurchase_relevants(
    catalog: Catalog,
    ldjson_path: Path,
) -> dict[int, set[int]]:
    name_to_group: dict[str, int] = {}
    for i, name in enumerate(catalog.product_names):
        key = _norm_name(name)
        if key and key not in name_to_group:
            name_to_group[key] = int(catalog.title_group[i])

    relevants: dict[int, set[int]] = {}
    with ldjson_path.open() as f:
        for line in f:
            row = json.loads(line)
            uid = row.get("uniq_id")
            if uid not in catalog.id_to_row:
                continue
            qrow = catalog.id_to_row[uid]
            raw = row.get("other_items_customers_buy") or ""
            if not isinstance(raw, str) or not raw.strip():
                continue
            groups: set[int] = set()
            for part in raw.split("|"):
                key = _norm_name(part)
                if not key:
                    continue
                g = name_to_group.get(key)
                if g is not None:
                    groups.add(g)
                else:
                    for nk, gv in name_to_group.items():
                        if key in nk or nk in key:
                            groups.add(gv)
                            break
            if groups:
                relevants[qrow] = groups
    return relevants


def split_title_groups(
    relevants: dict[int, set[int]],
    catalog: Catalog,
    seed: int = 1,
) -> tuple[set[int], set[int], set[int]]:
    all_groups = sorted({int(catalog.title_group[r]) for r in relevants})
    rng = random.Random(seed)
    rng.shuffle(all_groups)
    n = len(all_groups)
    t_end = int(0.6 * n)
    v_end = int(0.8 * n)
    train_g = set(all_groups[:t_end])
    val_g = set(all_groups[t_end:v_end])
    test_g = set(all_groups[v_end:])
    return train_g, val_g, test_g


def rows_for_groups(
    relevants: dict[int, set[int]], catalog: Catalog, groups: set[int]
) -> list[int]:
    return [r for r in relevants if int(catalog.title_group[r]) in groups]


def cap_rows(rows: list[int], max_q: int | None) -> list[int]:
    if max_q is None or len(rows) <= max_q:
        return rows
    rng = random.Random(0)
    pick = rows.copy()
    rng.shuffle(pick)
    return sorted(pick[:max_q])


def _top10_groups_full(
    catalog: Catalog,
    qrow: int,
    settings: Settings,
    weights: FusionWeights,
    profile: ScoringProfile,
) -> list[int]:
    ids = find_similar_products(
        catalog,
        catalog.uniq_ids[qrow],
        10,
        settings,
        weights=weights,
        profile=profile,
    )
    return [int(catalog.title_group[catalog.row_for(pid)]) for pid in ids]


def _same_child_share(catalog: Catalog, qrow: int, ids: list[str]) -> float:
    qc = int(catalog.category_code[qrow])
    if qc < 0 or not ids:
        return 0.0
    hit = sum(int(catalog.category_code[catalog.row_for(pid)] == qc) for pid in ids)
    return hit / len(ids)


def _same_family_share(catalog: Catalog, qrow: int, ids: list[str]) -> float:
    from service.cleaning import category_family_key

    qf = category_family_key(catalog.categories[qrow] or None)
    if not qf or not ids:
        return 0.0
    hit = sum(
        1
        for pid in ids
        if category_family_key(catalog.categories[catalog.row_for(pid)] or None) == qf
    )
    return hit / len(ids)


def _top10_ids(
    catalog: Catalog,
    qrow: int,
    settings: Settings,
    weights: FusionWeights,
    profile: ScoringProfile,
) -> list[str]:
    return find_similar_products(
        catalog,
        catalog.uniq_ids[qrow],
        10,
        settings,
        weights=weights,
        profile=profile,
    )


def rating_coverage_report(catalog: Catalog, sampled_pair_rate: float) -> dict[str, Any]:
    """Reconcile the Monte Carlo pair rate with row coverage. Scoring is unchanged."""
    n = int(catalog.n)
    pos = int(np.sum(catalog.rating_n > 0))
    row_share = pos / n if n else 0.0
    distinct = (pos * (pos - 1) / (n * (n - 1))) if n > 1 else 0.0
    with_replacement = row_share**2
    return {
        "rows": n,
        "rows_with_rating_n_gt_0": pos,
        "share_rows_rating_n_gt_0": row_share,
        "share_ordered_pairs_both_rating_n_gt_0_distinct": distinct,
        "share_ordered_pairs_both_rating_n_gt_0_with_replacement": with_replacement,
        "pair_active_rates_rating": sampled_pair_rate,
        "scoring": (
            "The rating block is active only when rating_n > 0 and the rating is finite "
            "for both the query and the candidate. rating_n == 0 is missing and the block "
            "is imputed to the calibrated 0.5. That rule was not changed."
        ),
        "why_pair_active_is_0_73": (
            "pair_active_rates.rating draws 5000 ordered pairs with replacement "
            "(the same row can be drawn twice). Its expectation is the square of the "
            f"row share ({row_share:.4f}^2 = {with_replacement:.4f}). The distinct-pair "
            f"share is {distinct:.4f}. The sampled value {sampled_pair_rate:.4f} matches "
            "that expectation, so the block is not active more often than the plan."
        ),
    }


def pair_active_rates(catalog: Catalog) -> dict[str, float]:
    n = catalog.n
    out: dict[str, float] = {"image": float(np.mean(catalog.image_mask))}
    for block, fn in [
        ("color", lambda i, j: not catalog.color_missing[i] and not catalog.color_missing[j]),
        (
            "price",
            lambda i, j: (
                np.isfinite(catalog.price[i])
                and np.isfinite(catalog.price[j])
                and catalog.price[i] > 0
                and catalog.price[j] > 0
            ),
        ),
        ("rating", lambda i, j: catalog.rating_n[i] > 0 and catalog.rating_n[j] > 0),
        (
            "weight",
            lambda i, j: (
                np.isfinite(catalog.weight_g[i])
                and np.isfinite(catalog.weight_g[j])
                and catalog.weight_g[i] > 0
                and catalog.weight_g[j] > 0
            ),
        ),
    ]:
        active = 0
        rng = random.Random(0)
        for _ in range(5000):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if fn(i, j):
                active += 1
        out[block] = active / 5000.0
    return out


def active_blocks_report(pair_rates: dict[str, float], image_coverage: float) -> dict[str, bool]:
    always = {b: True for b in ("text", "category", "audience", "brand")}
    always["image"] = image_coverage >= IMAGE_COVERAGE_FLOOR
    for b in ("color", "price", "rating", "weight"):
        always[b] = pair_rates.get(b, 0.0) > 0.05
    return always


def _latency_with_progress(
    catalog: Catalog, settings: Settings, n_queries: int = 500
) -> dict[str, float]:
    progress_log(f"latency: {n_queries} queries (lean + metadata full-catalog)")
    return latency_exact(catalog, settings, n_queries)


def latency_exact(
    catalog: Catalog,
    settings: Settings,
    n_queries: int = 500,
    warmup: int = 20,
) -> dict[str, Any]:
    rng = random.Random(0)
    draw = [rng.randrange(catalog.n) for _ in range(warmup + n_queries)]
    warm_rows, rows = draw[:warmup], draw[warmup:]
    w = FusionWeights(
        w=dict(DEFAULT_WEIGHTS),
        beta=1.0,
        text_image_transform=settings.text_image_transform,
    )
    catalog.active_profile = ScoringProfile()

    def _find(row: int) -> float:
        t0 = time.perf_counter()
        find_similar_products(catalog, catalog.uniq_ids[row], 10, settings, weights=w)
        return time.perf_counter() - t0

    def _explore(row: int) -> float:
        t0 = time.perf_counter()
        explore_similar(catalog, catalog.uniq_ids[row], 10, settings, w)
        return time.perf_counter() - t0

    for row in warm_rows:
        _find(row)
        _explore(row)
    lean = np.array([_find(row) for row in rows])
    explored = np.array([_explore(row) for row in rows])
    out: dict[str, Any] = {
        "find_similar_p50_ms": float(np.percentile(lean, 50) * 1000),
        "find_similar_p95_ms": float(np.percentile(lean, 95) * 1000),
        "explore_p50_ms": float(np.percentile(explored, 50) * 1000),
        "explore_p95_ms": float(np.percentile(explored, 95) * 1000),
        "p50_ms": float(np.percentile(lean, 50) * 1000),
        "p95_ms": float(np.percentile(lean, 95) * 1000),
        "warmup_queries": len(warm_rows),
        "n_queries": len(rows),
    }
    if out["find_similar_p50_ms"] > 100.0:
        totals = {name: 0.0 for name in BLOCK_NAMES}
        profile_rows = rows[:20]
        cand = np.arange(catalog.n, dtype=np.intp)
        for row in profile_rows:
            compute_block_batch(
                catalog,
                row,
                cand,
                w,
                metadata=False,
                block_timings=totals,
            )
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        out["block_profile_queries"] = len(profile_rows)
        out["block_profile_sec"] = {name: float(sec) for name, sec in ranked}
        out["block_profile_top3"] = [
            {
                "block": name,
                "sec": float(sec),
                "share": float(sec / max(sum(totals.values()), 1e-9)),
            }
            for name, sec in ranked[:3]
        ]
    return out


def _blazze_case(
    catalog: Catalog,
    settings: Settings,
    weights: FusionWeights,
    selected: str,
) -> dict[str, object]:
    needle = "modal v back g string"
    qrow = next(
        (i for i, name in enumerate(catalog.product_names) if needle in name.lower()),
        None,
    )
    if qrow is None:
        return {"error": "query not found"}
    binary_prof = ScoringProfile()
    catalog.active_profile = binary_prof
    before = explore_similar(
        catalog,
        catalog.uniq_ids[qrow],
        10,
        settings,
        weights,
    )
    prof = ScoringProfile.from_variant(selected)
    catalog.active_profile = prof
    after = explore_similar(
        catalog,
        catalog.uniq_ids[qrow],
        10,
        settings,
        weights,
    )
    catalog.active_profile = ScoringProfile()
    return {
        "query_id": catalog.uniq_ids[qrow],
        "binary_top10": before["results"],
        "selected_top10": after["results"],
        "ranking_changed": [r["id"] for r in before["results"]]
        != [r["id"] for r in after["results"]],
    }


def _metric_arrays_exact(
    rows: list[int],
    catalog: Catalog,
    settings: Settings,
    relevants: dict[int, set[int]],
    group_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    ndcg = np.full(len(rows), np.nan, dtype=np.float64)
    hit = np.full(len(rows), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        rel = relevants.get(r, set())
        if not rel:
            continue
        g = group_fn(r)
        ndcg[i] = ndcg_at_k(g, rel, 10)
        hit[i] = hit_at_k(g, rel, 10)
    return ndcg, hit


def _uncached_pool_sec_per_query(
    catalog: Catalog,
    rows: list[int],
    relevants: dict[int, set[int]],
) -> float:
    sample = rows[: min(40, len(rows))]
    if len(sample) < 5:
        return 0.0
    progress_log(f"uncached pool rate: {len(sample)} queries")
    t0 = time.perf_counter()
    PooledQueryStore.build(
        catalog,
        sample,
        relevants,
        TextImageTransform.TAIL_LOG,
        ScoringProfile(),
        1.0,
        inject_relevant_groups=True,
    )
    return (time.perf_counter() - t0) / len(sample)


def _extrapolated_full_run(
    stage_times: dict[str, float],
    *,
    sec_per_query: float,
    n_train_pools: int,
    n_eval_pools: int,
    tune_used: int,
    val_used: int,
    test_used: int,
    tune_full: int,
    val_full: int,
    test_full: int,
) -> dict[str, Any]:
    """Scale this run to the uncapped split. Pool time uses a fresh per-query build rate."""
    pool_sec = sec_per_query * (n_train_pools * tune_full + n_eval_pools * val_full)
    overhead = sum(v for k, v in stage_times.items() if k.endswith("_excluding_pool"))
    overhead_sec = overhead * (tune_full / max(tune_used, 1))
    exact_val = stage_times.get("exact_validation", 0.0) * (val_full / max(val_used, 1))
    exact_test = stage_times.get("exact_test", 0.0) * (test_full / max(test_used, 1))
    bootstrap = stage_times.get("bootstrap", 0.0) * (test_full / max(test_used, 1))
    shares = stage_times.get("variant_guardrail_shares", 0.0)
    share_sec = shares * ((val_full + test_full) / max(val_used + test_used, 1))
    latency = stage_times.get("latency", 0.0)
    total = pool_sec + overhead_sec + exact_val + exact_test + bootstrap + share_sec + latency
    return {
        "extrapolated_full_run_sec": total,
        "components_sec": {
            "pool_build": pool_sec,
            "tuning_excluding_pool": overhead_sec,
            "exact_validation": exact_val,
            "exact_test": exact_test,
            "bootstrap": bootstrap,
            "variant_guardrail_shares": share_sec,
            "latency": latency,
        },
        "uncached_pool_sec_per_query": sec_per_query,
        "train_pool_builds": n_train_pools,
        "eval_pool_builds": n_eval_pools,
        "note": (
            "Pool builds scale with uncapped query counts times the number of distinct "
            "cached pool configurations in this run. Tuning overhead, exact metrics, "
            "bootstrap, and variant share audits scale by the matching split. Latency "
            "stays at the 500-query benchmark."
        ),
    }


def main() -> None:
    _POOL_COUNTED.clear()
    t_start = time.perf_counter()
    progress_start()
    progress_log("starting")
    cap_tune, cap_val, cap_test = _split_caps(False)
    smoke = os.environ.get("EVAL_SMOKE", "").strip() in ("1", "true", "yes")
    legacy = os.environ.get("EVAL_MAX_QUERIES", "").strip()
    if legacy.isdigit():
        cap_tune = cap_val = cap_test = int(legacy)
        smoke = True
    if cap_tune is not None:
        smoke = True
    out_path = OUT_SMOKE if smoke else OUT
    tuned_out = TUNED_SMOKE_PATH if smoke else TUNED_PATH

    settings = Settings()
    artifacts = settings.artifacts_dir
    if not (artifacts / "manifest.json").exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"error": "artifacts missing; run make index"}, indent=2))
        print(f"Wrote {out_path}")
        return

    progress_log("loading catalog and co-purchase relevants")
    catalog = load_catalog(artifacts, build_hnsw=False)
    relevants = build_copurchase_relevants(catalog, LDJSON) if LDJSON.exists() else {}
    train_g, val_g, test_g = split_title_groups(relevants, catalog)
    ghash = _groups_hash(train_g, val_g, test_g)
    tune_all = rows_for_groups(relevants, catalog, train_g)
    val_all = rows_for_groups(relevants, catalog, val_g)
    test_all = rows_for_groups(relevants, catalog, test_g)
    tune_rows = cap_rows(tune_all, cap_tune)
    val_rows = cap_rows(val_all, cap_val)
    test_rows = cap_rows(test_all, cap_test)
    min_tune = int(os.environ.get("EVAL_TUNE_QUERIES", str(MIN_TUNE_QUERIES)))
    if not smoke and len(tune_rows) < min_tune:
        raise SystemExit(f"Need >={min_tune} train queries for tuning; got {len(tune_rows)}")
    split_info = {
        "train_groups": len(train_g),
        "val_groups": len(val_g),
        "test_groups": len(test_g),
        "hash": ghash,
        "tune_queries_full": len(tune_all),
        "validation_queries_full": len(val_all),
        "test_queries_full": len(test_all),
        "tune_queries": len(tune_rows),
        "validation_queries": len(val_rows),
        "test_queries": len(test_rows),
    }
    mode = "smoke" if smoke else "full"
    progress_log(f"{mode} eval: {split_info}")

    image_cov = float(catalog.manifest.get("image_coverage", 0.0) or 0.0)
    zero_image = image_cov < IMAGE_COVERAGE_FLOOR
    pair_rates = pair_active_rates(catalog)
    dedup_stats = dedup_merge_stats(
        [normalize_title_for_group(n) for n in catalog.product_names],
        catalog.brand_code.tolist(),
    )
    stage_times: dict[str, float] = {}
    bench = _run_microbenchmark(catalog, tune_rows, relevants, stage_times, ghash)
    bench["uncached_pool_sec_per_query"] = _uncached_pool_sec_per_query(
        catalog, tune_rows, relevants
    )
    if "pool_build_200_queries_sec" in bench:
        progress_log(
            f"benchmark pool_build={bench['pool_build_200_queries_sec']:.1f}s "
            f"weight_eval={bench['mean_weight_eval_sec']:.3f}s "
            f"bootstrap={bench['bootstrap_2000_resamples_sec']:.2f}s "
            f"uncached_pool_per_query={bench['uncached_pool_sec_per_query']:.4f}s"
        )

    val_gids = np.array([int(catalog.title_group[r]) for r in val_rows], dtype=np.int64)
    legacy_smoke_fw: FusionWeights | None = None
    if smoke and TUNED_SMOKE_PATH.exists():
        try:
            leg = json.loads(TUNED_SMOKE_PATH.read_text())
            legacy_smoke_fw = FusionWeights(
                w=leg["w"],
                beta=float(leg.get("beta", 1.0)),
                text_image_transform=TextImageTransform(
                    leg.get("text_image_transform", "tail_log")
                ),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            legacy_smoke_fw = None

    transform_val_pooled: dict[str, float] = {}
    val_transform_pq: dict[str, np.ndarray] = {}
    transform_tuned: dict[str, FusionWeights] = {}
    search_traces: dict[str, Any] = {}
    prof_bin = ScoringProfile()

    progress_log("phase 1/4: tune transforms (pooled train)")
    for transform in TextImageTransform:
        label = f"transform {transform.value}"
        t_phase = time.perf_counter()
        pool_before = stage_times.get("pool_build_this_run", 0.0)

        def make_store(
            beta: float,
            t: TextImageTransform = transform,
            p: ScoringProfile = prof_bin,
        ) -> PooledQueryStore:
            return _load_or_build_store(
                catalog, tune_rows, relevants, t, p, beta, True, ghash, stage_times
            )

        tuned, trace = tune_weights_hybrid(
            catalog,
            make_store,
            tune_rows,
            relevants,
            transform,
            zero_image,
            progress_label=label,
        )
        transform_tuned[transform.value] = tuned
        search_traces[transform.value] = trace
        val_store = _load_or_build_store(
            catalog,
            val_rows,
            relevants,
            transform,
            prof_bin,
            tuned.beta,
            False,
            ghash,
            stage_times,
        )
        mean_ndcg, arr = _pooled_val_ndcg(val_store, catalog, val_rows, relevants, tuned)
        transform_val_pooled[transform.value] = mean_ndcg
        val_transform_pq[transform.value] = arr
        wall = time.perf_counter() - t_phase
        spent_pool = stage_times.get("pool_build_this_run", 0.0) - pool_before
        stage_times[f"tuning_transform_{transform.value}"] = wall
        stage_times[f"tuning_transform_{transform.value}_excluding_pool"] = max(
            0.0, wall - spent_pool
        )

    shipped_key, transform_reason, transform_meta = select_transform_on_validation(
        val_transform_pq, val_gids
    )
    shipped = TextImageTransform(shipped_key)
    progress_log(f"shipped transform={shipped_key} ({transform_reason})")

    progress_log("phase 2/4: tune variants (pooled train)")
    variant_profiles: dict[str, ScoringProfile] = {
        "binary": ScoringProfile(),
        "A": ScoringProfile(category_mode="affinity"),
    }
    if catalog.text_nobrand is not None:
        variant_profiles["B"] = ScoringProfile(text_matrix="nobrand")
        variant_profiles["A+B"] = ScoringProfile(category_mode="affinity", text_matrix="nobrand")

    val_ndcg_pq: dict[str, np.ndarray] = {}
    tuned_by_variant: dict[str, FusionWeights] = {}
    variant_traces: dict[str, Any] = {}
    for vname, vprof in variant_profiles.items():
        t_phase = time.perf_counter()
        pool_before = stage_times.get("pool_build_this_run", 0.0)

        def make_store_v(
            beta: float,
            p: ScoringProfile = vprof,
        ) -> PooledQueryStore:
            return _load_or_build_store(
                catalog, tune_rows, relevants, shipped, p, beta, True, ghash, stage_times
            )

        tw, vtrace = tune_weights_hybrid(
            catalog,
            make_store_v,
            tune_rows,
            relevants,
            shipped,
            zero_image,
            progress_label=f"variant {vname}",
        )
        tuned_by_variant[vname] = tw
        variant_traces[vname] = vtrace
        val_store = _load_or_build_store(
            catalog, val_rows, relevants, shipped, vprof, tw.beta, False, ghash, stage_times
        )
        _, arr = _pooled_val_ndcg(val_store, catalog, val_rows, relevants, tw)
        val_ndcg_pq[vname] = arr
        wall = time.perf_counter() - t_phase
        spent_pool = stage_times.get("pool_build_this_run", 0.0) - pool_before
        stage_times[f"tuning_variant_{vname}"] = wall
        stage_times[f"tuning_variant_{vname}_excluding_pool"] = max(0.0, wall - spent_pool)

    selected, selection_reason, selection_meta = select_variant_on_validation(val_ndcg_pq, val_gids)
    sel_prof = variant_profiles[selected]
    tuned_candidate = tuned_by_variant[selected]
    progress_log(f"selected variant={selected} ({selection_reason})")

    default_fw = FusionWeights(
        w=dict(DEFAULT_WEIGHTS),
        beta=1.0,
        text_image_transform=shipped,
    )
    if zero_image:
        default_fw = FusionWeights(
            w={**dict(DEFAULT_WEIGHTS), "image": 0.0},
            beta=1.0,
            text_image_transform=shipped,
        )
    guard_val_default = mean_guardrails(catalog, val_rows, settings, default_fw, sel_prof)
    guard_val_tuned = mean_guardrails(catalog, val_rows, settings, tuned_candidate, sel_prof)
    g_ok, g_fail = guardrails_pass(guard_val_default, guard_val_tuned)
    guardrail_info: dict[str, Any] = {
        "validation_default": guard_val_default,
        "validation_tuned_candidate": guard_val_tuned,
        "accepted_tuned": g_ok,
        "rejection_reason": g_fail,
    }
    sel_weights = tuned_candidate if g_ok else default_fw
    if not g_ok:
        progress_log(f"guardrails: using default weights ({g_fail})")

    progress_log("variant shares: binary vs A")
    t_shares = time.perf_counter()
    binary_fw = tuned_by_variant["binary"]
    affinity_fw = tuned_by_variant["A"]
    binary_prof = ScoringProfile()
    affinity_prof = ScoringProfile(category_mode="affinity")

    def _variant_share_block(rows: list[int], gids: np.ndarray) -> dict[str, Any]:
        b_arr = guardrail_query_arrays(catalog, rows, settings, binary_fw, binary_prof)
        a_arr = guardrail_query_arrays(catalog, rows, settings, affinity_fw, affinity_prof)
        block: dict[str, Any] = {}
        for key in b_arr:
            block[key] = {
                "binary_mean": float(np.nanmean(b_arr[key])),
                "A_mean": float(np.nanmean(a_arr[key])),
                "paired_A_minus_binary": paired_bootstrap_ci(a_arr[key], b_arr[key], gids),
            }
        return block

    variant_guardrails = {
        "weights": "each variant's tuned weights; selection rule unchanged",
        "validation": _variant_share_block(val_rows, val_gids),
        "test": _variant_share_block(
            test_rows, np.array([int(catalog.title_group[r]) for r in test_rows], dtype=np.int64)
        ),
    }
    stage_times["variant_guardrail_shares"] = time.perf_counter() - t_shares

    progress_log("phase 3/4: exact validation + test metrics (precompute + bootstrap)")
    val_store_final = _load_or_build_store(
        catalog, val_rows, relevants, shipped, sel_prof, sel_weights.beta, False, ghash, stage_times
    )
    store_map = {
        r: (val_store_final.by_row[r].pool, val_store_final.by_row[r].blocks)
        for r in val_rows
        if r in val_store_final.by_row
    }
    pool_fidelity = audit_pool_fidelity(
        catalog, settings, val_rows, sel_weights, sel_prof, store_map
    )
    rel_cov = _relevant_coverage(val_store_final, val_rows)

    t_exact_val = time.perf_counter()

    def top_full(c: Catalog, r: int, w: FusionWeights, p: ScoringProfile) -> list[int]:
        return _top10_groups_full(c, r, settings, w, p)

    validation_exact = {
        "ndcg10": mean_metrics_on_rows(
            catalog,
            val_rows,
            relevants,
            sel_weights,
            sel_prof,
            lambda g, rel: ndcg_at_k(g, rel, 10),
            top_full,
        ),
        "hit10": mean_metrics_on_rows(
            catalog,
            val_rows,
            relevants,
            sel_weights,
            sel_prof,
            lambda g, rel: hit_at_k(g, rel, 10),
            top_full,
        ),
    }
    val_variant_pooled = {k: float(np.nanmean(v)) for k, v in val_ndcg_pq.items()}

    val_fusion_ndcg, _ = _metric_arrays_exact(
        val_rows,
        catalog,
        settings,
        relevants,
        lambda r: _top10_groups_full(catalog, r, settings, sel_weights, sel_prof),
    )
    val_minilm_ndcg, _ = _metric_arrays_exact(
        val_rows,
        catalog,
        settings,
        relevants,
        lambda r: _top10_groups_minilm(catalog, r),
    )
    stage_times["exact_validation"] = time.perf_counter() - t_exact_val
    t_boot = time.perf_counter()
    paired_val_fusion_minilm = paired_bootstrap_ci(val_fusion_ndcg, val_minilm_ndcg, val_gids)
    stage_times["bootstrap"] = time.perf_counter() - t_boot

    test_gids = np.array([int(catalog.title_group[r]) for r in test_rows], dtype=np.int64)
    t_exact_test = time.perf_counter()
    fusion_ndcg, fusion_hit = _metric_arrays_exact(
        test_rows,
        catalog,
        settings,
        relevants,
        lambda r: _top10_groups_full(catalog, r, settings, sel_weights, sel_prof),
    )
    stage_times["exact_test"] = time.perf_counter() - t_exact_test
    progress_log("bootstrap: fusion test")
    t_boot = time.perf_counter()
    test_fusion = {
        "ndcg10": cluster_bootstrap_ci(fusion_ndcg, test_gids),
        "hit10": cluster_bootstrap_ci(fusion_hit, test_gids),
    }
    stage_times["bootstrap"] = stage_times.get("bootstrap", 0.0) + (time.perf_counter() - t_boot)
    baselines: dict[str, Any] = {}
    minilm_test_ndcg = np.full(len(test_rows), np.nan, dtype=np.float64)
    for name, fn in [
        ("popularity", lambda r: _top10_groups_popularity(catalog, r)),
        ("minilm_only", lambda r: _top10_groups_minilm(catalog, r)),
    ]:
        t_exact_test = time.perf_counter()
        b_ndcg, b_hit = _metric_arrays_exact(test_rows, catalog, settings, relevants, fn)
        stage_times["exact_test"] = stage_times.get("exact_test", 0.0) + (
            time.perf_counter() - t_exact_test
        )
        if name == "minilm_only":
            minilm_test_ndcg = b_ndcg
        t_boot = time.perf_counter()
        baselines[name] = {
            "ndcg10": cluster_bootstrap_ci(b_ndcg, test_gids),
            "hit10": cluster_bootstrap_ci(b_hit, test_gids),
        }
        stage_times["bootstrap"] = stage_times.get("bootstrap", 0.0) + (
            time.perf_counter() - t_boot
        )
    t_boot = time.perf_counter()
    paired_test_fusion_minilm = paired_bootstrap_ci(
        fusion_ndcg,
        minilm_test_ndcg,
        test_gids,
    )
    stage_times["bootstrap"] = stage_times.get("bootstrap", 0.0) + (time.perf_counter() - t_boot)

    guard_test_default = mean_guardrails(catalog, test_rows, settings, default_fw, sel_prof)
    guard_test_shipped = mean_guardrails(catalog, test_rows, settings, sel_weights, sel_prof)

    progress_log("phase 4/4: latency and Blazze")
    t_lat = time.perf_counter()
    latency_report = _latency_with_progress(catalog, settings, 500)
    stage_times["latency"] = time.perf_counter() - t_lat
    ship_trace = search_traces.get(shipped.value, [])
    runtime = time.perf_counter() - t_start
    stage_summary = {
        "pool_build": stage_times.get("pool_build", 0.0),
        "pool_build_train": stage_times.get("pool_build_train", 0.0),
        "pool_build_eval": stage_times.get("pool_build_eval", 0.0),
        "pool_build_this_run": stage_times.get("pool_build_this_run", 0.0),
        "exact_validation": stage_times.get("exact_validation", 0.0),
        "exact_test": stage_times.get("exact_test", 0.0),
        "bootstrap": stage_times.get("bootstrap", 0.0),
        "variant_guardrail_shares": stage_times.get("variant_guardrail_shares", 0.0),
        "latency": stage_times.get("latency", 0.0),
    }
    for key, value in stage_times.items():
        if key.startswith("tuning_transform_") or key.startswith("tuning_variant_"):
            stage_summary[key] = value
    stage_summary["total"] = runtime
    stage_times["total"] = runtime
    print_runtime_table(stage_summary)
    n_train_pools = sum(1 for name in _POOL_COUNTED if name.startswith("pool_train_"))
    n_eval_pools = sum(1 for name in _POOL_COUNTED if name.startswith("pool_eval_"))
    extrapolated = _extrapolated_full_run(
        stage_times,
        sec_per_query=float(bench.get("uncached_pool_sec_per_query", 0.0)),
        n_train_pools=n_train_pools,
        n_eval_pools=n_eval_pools,
        tune_used=len(tune_rows),
        val_used=len(val_rows),
        test_used=len(test_rows),
        tune_full=len(tune_all),
        val_full=len(val_all),
        test_full=len(test_all),
    )

    report: dict[str, Any] = {
        "smoke": smoke,
        "fast_eval": False,
        "fast_eval_note": (
            "fast_eval is false. EVAL_SMOKE only caps tune, validation, and test query "
            "counts. Selection, guardrails, exact metrics, and cluster bootstrap use the "
            "same path as a full eval."
        ),
        "split": split_info,
        "run": {
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime_sec": runtime,
            "runtime_stages_sec": stage_summary,
            "extrapolated_full_run": extrapolated,
            "smoke": smoke,
            "benchmark": bench,
            "config": {
                "search_mode": settings.search_mode,
                "eval_tune_queries_min": min_tune,
                "n_bootstrap": N_BOOT_DEFAULT,
            },
        },
        "row_count": catalog.n,
        "image_coverage": image_cov,
        "image_weight_zeroed": zero_image,
        "image_weight_note": (
            "Image weight stays 0 because image_coverage is under 5%."
            if zero_image
            else "Image coverage is at least 5%; the image block is eligible."
        ),
        "active_blocks": active_blocks_report(pair_rates, image_cov),
        "mu_transform": catalog.manifest.get("mu_transform"),
        "shipped_transform": shipped.value,
        "transform_selection_reason": transform_reason,
        "transform_selection_meta": transform_meta,
        "selected_variant": selected,
        "selection_reason": selection_reason,
        "selection_meta": selection_meta,
        "guardrails": guardrail_info,
        "guardrails_test": {
            "default": guard_test_default,
            "shipped": guard_test_shipped,
        },
        "variant_guardrails_binary_vs_A": variant_guardrails,
        "paired_ndcg_fusion_minus_minilm": {
            "validation": paired_val_fusion_minilm,
            "test": paired_test_fusion_minilm,
        },
        "rating_pair_active_rate": pair_rates.get("rating"),
        "rating_coverage": rating_coverage_report(catalog, float(pair_rates.get("rating", 0.0))),
        "rating_missing_when_n_zero": (
            "rating block active only when rating_n>0 for query and candidate; "
            "otherwise imputed to 0.5 calibrated"
        ),
        "title_group_top10": top_title_groups(catalog, 10),
        "validation_transform_ndcg10_pooled": transform_val_pooled,
        "validation_variant_ndcg10_pooled": val_variant_pooled,
        "validation_exact": validation_exact,
        "pool_fidelity": pool_fidelity,
        "relevant_groups_in_pool_mean": rel_cov,
        "weight_search": {
            "search_trace": {
                **search_traces,
                **{f"variant_{k}": v for k, v in variant_traces.items()},
            },
            "unchanged_reason": weights_unchanged_reason(ship_trace),
        },
        "test_results": {"fusion": test_fusion, "baselines": baselines},
        "tune_queries": len(tune_rows),
        "validation_queries": len(val_rows),
        "test_queries": len(test_rows),
        "latency_exact": latency_report,
        "pair_active_rates": pair_rates,
        "title_group_hist": {"unique_groups": int(len(np.unique(catalog.title_group)))},
        "dedup_merge": dedup_stats,
        "cases": {
            "blazze": _blazze_case(catalog, settings, sel_weights, selected),
            "blazze_weights_compare": blazze_comparison_table(
                catalog,
                settings,
                default_fw=default_fw,
                legacy_smoke_fw=legacy_smoke_fw,
                selected_fw=sel_weights,
                selected_variant=selected,
                variant_a_fw=affinity_fw,
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    _write_tuned_weights(
        tuned_out,
        sel_weights,
        selected,
        "smoke" if smoke else "full",
        split_info,
        selection_reason=selection_reason,
        transform_reason=transform_reason,
        guardrails=guardrail_info,
    )
    progress_log(f"finished in {runtime:.1f}s — wrote {out_path} and {tuned_out}")
    print(f"Wrote {out_path}")

    if smoke:
        return

    render_design(report, DESIGN_PATH)
    render_current_state_markers(report, DESIGN_PATH)
    render_current_state_markers(report, SUMMARY_PATH)
    print(f"Updated {DESIGN_PATH}, {SUMMARY_PATH}")


def _top10_groups_minilm(catalog: Catalog, qrow: int) -> list[int]:
    scores = catalog.text @ catalog.text[qrow]
    scores[qrow] = -np.inf
    qg = int(catalog.title_group[qrow])
    order = np.argsort(-scores)
    picked: list[int] = []
    seen: set[int] = set()
    for idx in order:
        if len(picked) >= 10:
            break
        g = int(catalog.title_group[idx])
        if g == qg or g in seen:
            continue
        picked.append(g)
        seen.add(g)
    return picked


def _top10_groups_popularity(catalog: Catalog, qrow: int) -> list[int]:
    qc = int(catalog.category_code[qrow])
    mask = catalog.category_code == qc
    scores = np.where(mask, catalog.rating_n.astype(np.float64), -1.0)
    scores[qrow] = -1.0
    order = np.argsort(-scores)
    picked: list[int] = []
    seen: set[int] = set()
    qg = int(catalog.title_group[qrow])
    for idx in order:
        if len(picked) >= 10:
            break
        if scores[idx] < 0:
            continue
        g = int(catalog.title_group[idx])
        if g == qg or g in seen:
            continue
        picked.append(g)
        seen.add(g)
    return picked


if __name__ == "__main__":
    main()
