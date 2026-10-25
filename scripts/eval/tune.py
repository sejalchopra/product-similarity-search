"""Weight search helpers for evaluate.py."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from scripts.eval.pooled import PooledQueryStore, top10_groups_on_pool
from scripts.eval.progress import log as progress_log
from service.catalog import Catalog
from service.scoring import BLOCK_NAMES, DEFAULT_WEIGHTS, FusionWeights
from service.scoring_profile import ScoringProfile
from service.transforms import TextImageTransform

MULTIPLIERS = (0.5, 2.0)
DIRICHLET_DRAWS = 40
DIRICHLET_CONC = 30.0
MAX_TEXT_WEIGHT = 0.5
MAX_BLOCK_MULT = 3.0


def ndcg_at_k(ranked_groups: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, g in enumerate(ranked_groups[:k]):
        if g in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_at_k(ranked_groups: list[int], relevant: set[int], k: int = 10) -> float:
    return float(any(g in relevant for g in ranked_groups[:k]))


def _vec_to_w(vec: np.ndarray) -> dict[str, float]:
    return {b: float(vec[i]) for i, b in enumerate(BLOCK_NAMES)}


def _apply_weight_constraints(
    vec: np.ndarray,
    start: np.ndarray,
    zero_image: bool,
) -> np.ndarray | None:
    v = vec.astype(np.float64).copy()
    if zero_image:
        v[1] = 0.0
    for i in range(len(v)):
        v[i] = min(v[i], start[i] * MAX_BLOCK_MULT)
        if v[i] < 0:
            return None
    s = float(v.sum())
    if s <= 0:
        return None
    norm = v / s
    if float(norm[0]) > MAX_TEXT_WEIGHT:
        return None
    return v


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    s = float(vec.sum())
    if s <= 0:
        return vec
    return vec / s


class TuneRunner:
    """Evaluate weight dicts via matvec on a pooled block cache."""

    def __init__(
        self,
        catalog: Catalog,
        store: PooledQueryStore,
        tune_rows: list[int],
        relevants: dict[int, set[int]],
        transform: TextImageTransform,
        beta: float,
        zero_image: bool,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.tune_rows = tune_rows
        self.relevants = relevants
        self.transform = transform
        self.beta = beta
        self.zero_image = zero_image

    def _fw(self, w: dict[str, float]) -> FusionWeights:
        ww = dict(w)
        if self.zero_image:
            ww["image"] = 0.0
        return FusionWeights(w=ww, beta=self.beta, text_image_transform=self.transform)

    def eval_weights(self, w: dict[str, float]) -> tuple[float, float]:
        fw = self._fw(w)
        ndcgs: list[float] = []
        hits: list[float] = []
        for qrow in self.tune_rows:
            rel = self.relevants.get(qrow, set())
            if not rel:
                continue
            ent = self.store.by_row[qrow]
            groups = top10_groups_on_pool(self.catalog, qrow, ent.pool, ent.blocks, fw)
            ndcgs.append(ndcg_at_k(groups, rel, 10))
            hits.append(hit_at_k(groups, rel, 10))
        if not ndcgs:
            return 0.0, 0.0
        return float(np.mean(ndcgs)), float(np.mean(hits))


def tune_weights_hybrid(
    catalog: Catalog,
    make_store: Callable[[float], PooledQueryStore],
    tune_rows: list[int],
    relevants: dict[int, set[int]],
    transform: TextImageTransform,
    zero_image: bool,
    *,
    progress_label: str = "",
) -> tuple[FusionWeights, list[dict[str, Any]]]:
    if progress_label:
        progress_log(f"{progress_label}: weight search ({len(tune_rows)} train queries)")
    trace: list[dict[str, Any]] = []
    start_vec = np.array([DEFAULT_WEIGHTS[b] for b in BLOCK_NAMES], dtype=np.float64)
    if zero_image:
        start_vec[1] = 0.0
    best_vec = start_vec.copy()
    best_beta = 1.0
    stop_reason = "completed search"
    runner = TuneRunner(catalog, make_store(1.0), tune_rows, relevants, transform, 1.0, zero_image)
    start_w = {b: float(best_vec[i]) for i, b in enumerate(BLOCK_NAMES)}
    best_ndcg, best_hit = runner.eval_weights(start_w)
    trace.append(
        {
            "weights": start_w,
            "beta": best_beta,
            "tune_ndcg10": best_ndcg,
            "tune_hit10": best_hit,
            "accepted": True,
            "note": "start",
        }
    )

    rng = np.random.default_rng(42)
    w0 = _normalize_vec(best_vec)
    for _ in range(DIRICHLET_DRAWS):
        alpha = w0 * DIRICHLET_CONC + 1e-6
        trial = rng.dirichlet(alpha)
        constrained = _apply_weight_constraints(trial, start_vec, zero_image)
        if constrained is None:
            continue
        tw = _vec_to_w(constrained)
        ndcg, hit = runner.eval_weights(tw)
        accepted = ndcg > best_ndcg or (ndcg == best_ndcg and hit > best_hit)
        trace.append(
            {
                "weights": tw,
                "beta": best_beta,
                "tune_ndcg10": ndcg,
                "tune_hit10": hit,
                "accepted": accepted,
                "note": "dirichlet",
            }
        )
        if accepted:
            best_vec = constrained.copy()
            best_ndcg, best_hit = ndcg, hit

    for trial_beta in (0.5, 1.0):
        runner = TuneRunner(
            catalog,
            make_store(trial_beta),
            tune_rows,
            relevants,
            transform,
            trial_beta,
            zero_image,
        )
        vec = best_vec.copy()
        cur_w = {b: float(vec[i]) for i, b in enumerate(BLOCK_NAMES)}
        cur_ndcg, cur_hit = runner.eval_weights(cur_w)
        for _pass in range(2):
            improved = False
            for bi, block in enumerate(BLOCK_NAMES):
                for mult in MULTIPLIERS:
                    trial = vec.copy()
                    trial[bi] = trial[bi] * mult
                    constrained = _apply_weight_constraints(trial, start_vec, zero_image)
                    if constrained is None:
                        continue
                    tw = _vec_to_w(constrained)
                    ndcg, hit = runner.eval_weights(tw)
                    accepted = ndcg > cur_ndcg or (ndcg == cur_ndcg and hit > cur_hit)
                    trace.append(
                        {
                            "weights": tw,
                            "beta": trial_beta,
                            "tune_ndcg10": ndcg,
                            "tune_hit10": hit,
                            "accepted": accepted,
                            "block": block,
                            "multiplier": mult,
                            "note": "coordinate",
                        }
                    )
                    if accepted:
                        vec = constrained
                        cur_ndcg, cur_hit = ndcg, hit
                        improved = True
            if not improved:
                stop_reason = f"coordinate pass stalled at beta={trial_beta}"
                break
        if cur_ndcg > best_ndcg or (cur_ndcg == best_ndcg and cur_hit > best_hit):
            best_vec = vec
            best_beta = trial_beta
            best_ndcg, best_hit = cur_ndcg, cur_hit

    final_w = {b: float(best_vec[i]) for i, b in enumerate(BLOCK_NAMES)}
    if zero_image:
        final_w["image"] = 0.0
    final = FusionWeights(
        w=final_w,
        beta=best_beta,
        text_image_transform=transform,
    )
    if progress_label:
        progress_log(
            f"{progress_label}: done NDCG@10={best_ndcg:.4f} hit@10={best_hit:.4f} beta={best_beta}"
        )
    trace.append(
        {
            "weights": final.w,
            "beta": final.beta,
            "tune_ndcg10": best_ndcg,
            "tune_hit10": best_hit,
            "accepted": True,
            "note": "final",
        }
    )
    trace.append({"search_stop_reason": stop_reason})
    return final, trace


def weights_unchanged_reason(trace: list[dict[str, Any]]) -> str | None:
    weight_entries = [e for e in trace if "weights" in e]
    if len(weight_entries) < 2:
        return None
    start = weight_entries[0]["weights"]
    final = weight_entries[-1]["weights"]
    if start == final:
        return "no coordinate step beat the starting tune NDCG@10"
    return None


def mean_metrics_on_rows(
    catalog: Catalog,
    rows: list[int],
    relevants: dict[int, set[int]],
    fw: FusionWeights,
    profile: ScoringProfile,
    metric_fn: Callable[[list[int], set[int]], float],
    top_fn: Callable[[Catalog, int, FusionWeights, ScoringProfile], list[int]],
) -> float:
    vals: list[float] = []
    for qrow in rows:
        rel = relevants.get(qrow, set())
        if not rel:
            continue
        groups = top_fn(catalog, qrow, fw, profile)
        vals.append(metric_fn(groups, rel))
    return float(np.mean(vals)) if vals else 0.0
