# Design note

All metrics below are copied from [`eval.json`](eval.json) (`make eval`). Do not hand-edit numbers in the eval region.

**Eval tiers:** `make eval-smoke`, `make eval-medium` (default capped query env vars in the Makefile), and `make eval` share one pipeline. Smoke and medium write only to gitignored `docs/eval.smoke.json` and `docs/eval.medium.json`; they do not overwrite `docs/eval.json`. Pool stages resume from `artifacts/eval_cache/`. Only `make eval` refreshes the generated blocks in this file.

<!-- current_state:start -->
## Current state (from eval.json)

_Run `make eval` to populate this block from `docs/eval.json`._
<!-- current_state:end -->

<!-- eval:start -->
## Problem and data

See `eval.json` after `make eval`.
<!-- eval:end -->

## Problem and messy catalog (narrative)

The Amazon fashion sample is usable for a prototype but noisy for learning and evaluation. Field formats are inconsistent (prices, weights, ratings), brands and categories are often missing or generic, and many listings are near-duplicate title variants (color/size) that should not all appear as separate “similar” hits. Image URLs exist for most rows, but crawl failures and placeholders still matter for multimodal blocks. The index build scripts record coverage, deduplication, and cleaning statistics in `artifacts/manifest.json` and related reports; the eval JSON summarizes how those choices affect block activity and retrieval quality.

## Nine-block late fusion

Similarity is a **weighted sum of nine blocks** (text, image, category, audience, brand, color, price, rating, weight), each producing a value in \([0,1]\) after calibration—not a single cosine on a concatenated vector.

**Why separate blocks:** Attributes differ in type, missingness, and scale. A single embedding cannot enforce “same audience” or cap categorical noise when brand is absent. Blocks let us impute missing pieces independently and tune or zero weights when a signal is unreliable.

**Tail-log calibration:** Raw and percentile transforms were compared on validation (see `validation_transform_ndcg10_pooled` in eval JSON). **Tail-log** on text/image cosines spreads mass in the high-similarity tail where neighbors matter for ranking, without the distortion we saw from percentile binning on sparse tails.

**Expected-value imputation:** When a block is missing for query or candidate, we use a **table-conditional expectation** (category, audience, etc.) estimated on the catalog, then apply the same tail-log mapping. That avoids treating “unknown” as neutral 0.5 when the marginal is skewed.

**Title-group dedup:** Normalized titles define groups; retrieval and evaluation operate on **one representative per group** so colorway siblings do not dominate top-10 lists. Merge stats appear under `dedup_merge` in eval JSON.

**Capped categorical missing values:** High-cardinality codes with weak support use **capped** imputation so rare levels do not act like strong matches; brand backfill and category affinity tables are built at index time (`artifacts/`).

## Evaluation design

**Splits:** Train / validation / test are **disjoint by title group**, not by row, so the same product family cannot leak across splits. Group counts and query counts per split are in `split` on the eval report.

**Labels:** Relevance is a **co-purchase proxy**: other SKUs whose normalized title appears in a product’s “also bought” text field. This is cheap on the ldjson but **noisy** (string matching, not ASIN graph, incomplete co-buy fields). Treat offline NDCG as directional.

**Baselines:** MiniLM text-only and popularity-style category baselines are scored with the same pooled exact machinery as fusion.

**Paired cluster bootstrap:** Validation and test report **paired** fusion vs MiniLM differences with cluster bootstrap CIs (`paired_ndcg_fusion_minus_minilm`).

**Guardrails:** Tuned weights must not collapse demographic/category agreement shares vs defaults (`guardrails`, `variant_guardrails_binary_vs_A`). Failed guardrails **reject** the tuned vector even if validation NDCG moves slightly up.

## Findings (qualitative; numbers in eval JSON)

- **Fusion vs text-only:** On validation and test, fusion is intended to beat MiniLM-only when tabular blocks carry signal; see paired CI fields in the generated eval block and `test_results`.
- **Percentile vs tail-log:** Percentile calibration underperformed tail-log on validation pooled NDCG; transform selection reason is in `transform_selection_reason` / `transform_selection_meta`.
- **Tuned weights:** Search often nudges weights, but **guardrails frequently reject** the tuned vector; shipped weights remain the hand-set defaults in `artifacts/tuned_weights.json` (`unchanged_reason` in `weight_search`).
- **Category-affinity variant (A):** Variant **A** (category affinity table) is compared to binary category on validation; selection metadata documents whether it ships (`selected_variant`, `selection_reason`).
- **Blazze-type queries:** Long, attribute-heavy titles stress text and imputation; case studies and weight comparisons are under `cases.blazze` and `cases.blazze_weights_compare` (see also [blazze-ranking.md](blazze-ranking.md)).

## Latency (exact path)

Latency numbers are under `latency_exact` in eval JSON (500-query benchmark on loaded artifacts). They contrast lean `/find_similar_products` with full-catalog scoring paths. In-process LRU caching backs repeated `/find_similar_products` calls; do not run heavy jobs alongside eval if you want a clean latency stage measurement.

## ANN / HNSW

- **Online:** `SEARCH_MODE=ann` uses a **capped union** (text/HNSW top‑K ∪ same-brand ∪ same-category lists), then the **same fusion scorer** as exact mode.
- **Bench:** `make bench` measures **pure HNSW** on synthetic 1M×384 vectors (`docs/bench_hnsw.json`); union retrieval at 1M catalog rows is **not** benchmarked there.
- **Reference:** Yu. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020 (arXiv:1603.09320).

## Deploy

- `docker build` copies `artifacts/` and `docs/eval.json` (no runtime ldjson).
- [`k8s.yaml`](../k8s.yaml): Deployment + Service, probes, `imagePullPolicy: IfNotPresent`.
- **Stretch:** `docker-compose.yml` + `scripts/load_qdrant.py` (Qdrant named vectors).

## Limitations

- **Image coverage** is high in this crawl but not complete; whether the image block is zeroed is recorded in eval JSON (`image_coverage`, `image_weight_zeroed`).
- **Co-purchase labels** remain a weak proxy for true relevance.
- **Sparse numeric fields** (e.g. weight) limit some blocks’ pair-active rates (`pair_active_rates` in eval JSON).
- **ANN union** trades recall for latency; exact mode is the reference for assignment correctness.
