# Design note

All metrics below are copied from [`eval.json`](eval.json) (`make eval`). Do not hand-edit numbers in the eval region.

**Eval tiers:** `make eval-smoke` and `make eval-medium` write gitignored sidecar JSON; promoting a capped run copies into `docs/eval.json` (current file). `make eval` runs uncapped and refreshes the generated blocks below. Pool stages resume from `artifacts/eval_cache/`.

<!-- current_state:start -->
## Current state (from eval.json)

- **Generated (UTC):** 2026-10-25T10:34:28Z
- **Git SHA:** 6b5050eda342720c67c4748965fada6e38831c01
- **Eval mode:** capped (not smoke)
- **Split hash:** e400213588923827
- **Queries (used / available):** tune 1500 / 12660, validation 1500 / 4224, test 2000 / 4135
- **Selected variant:** binary
- **Shipped transform:** tail_log
- **Image coverage:** 0.9953666666666666
- **Image weight in tuning:** active
- **Latency /find_similar_products (lean):** p50 166.73516645096242 ms, p95 173.74844231526365 ms
- **Latency full-catalog w/ metadata:** p50 — ms, p95 — ms
- **Active blocks (pair-active or coverage):** {'text': True, 'category': True, 'audience': True, 'brand': True, 'image': True, 'color': True, 'price': True, 'rating': True, 'weight': False}

<!-- current_state:end -->

<!-- eval:start -->
## Problem and data (from eval.json)

- **Eval run:** capped (not smoke). Query caps: tune **1500** / 12660, validation **1500** / 4224, test **2000** / 4135.
- **Catalog:** 30000 rows.
- **Image coverage:** 0.9953666666666666.
- **Title groups:** 21627 unique keys for variant dedup.
- **Splits:** tune 1500, validation 1500, test 2000 queries.

## Scoring

Shipped transform: **tail_log** (chosen on validation). Selected variant: **binary** — only binary variant available or binary won outright.

Weight search: search moved weights (see eval.json → weight_search.search_trace).

Image block in tuning: active.

## Test evaluation (co-purchase @ title_group)

Metrics below are computed **once** on the held-out test split after selection. Cluster-bootstrap 95% CIs.

| Method | NDCG@10 | Hit@10 |
|--------|---------|--------|
| Fusion (selected) | 0.284 [0.268, 0.302] | 0.603 [0.574, 0.631] |
| MiniLM-only | 0.248 [0.232, 0.265] | 0.556 [0.527, 0.586] |
| Popularity (same child) | 0.052 [0.045, 0.060] | 0.157 [0.139, 0.175] |

Validation scores for transforms and variants: `eval.json` → `validation_transform_ndcg10_pooled`, `validation_variant_ndcg10_pooled`.

Pair-active rates: color 0.2382, price 0.8212, rating 0.73, weight 0.0434, image 0.9953666666666666.

Dedup merge: {'missing_brand_rows_reassigned': 16, 'largest_title_groups': [{'title_group': 99, 'row_count': 81}, {'title_group': 89, 'row_count': 70}, {'title_group': 238, 'row_count': 63}, {'title_group': 202, 'row_count': 55}, {'title_group': 182, 'row_count': 48}, {'title_group': 284, 'row_count': 48}, {'title_group': 52, 'row_count': 43}, {'title_group': 320, 'row_count': 43}, {'title_group': 3274, 'row_count': 43}, {'title_group': 1921, 'row_count': 41}, {'title_group': 80, 'row_count': 40}, {'title_group': 1101, 'row_count': 39}, {'title_group': 369, 'row_count': 37}, {'title_group': 1452, 'row_count': 36}, {'title_group': 509, 'row_count': 35}, {'title_group': 527, 'row_count': 35}, {'title_group': 548, 'row_count': 35}, {'title_group': 763, 'row_count': 34}, {'title_group': 44, 'row_count': 33}, {'title_group': 554, 'row_count': 33}]}.

## Blazze before / after (generated)

| Rank | Before (binary) | After (selected) |
|------|-----------------|------------------|
| 1 | The Blazze Men's Modal G String Sexy Low Rise Briefs Panties, Men Boxer Underpants Shorts Underwear Bulge Pouch Funny Thongs (0.5615524793825368) | The Blazze Men's Modal G String Sexy Low Rise Briefs Panties, Men Boxer Underpants Shorts Underwear Bulge Pouch Funny Thongs (0.5615524793825368) |
| 2 | La Intimo Men’s Nylon Spandex Lace Thong Underwear (0.4643356359569048) | La Intimo Men’s Nylon Spandex Lace Thong Underwear (0.4643356359569048) |
| 3 | The Blazze 1013 Women's Thong High Rise Sexy Solid G-String Thong Bikini T-String Sexy Lingerie Hipsters Panties Briefs (0.4214149595505697) | The Blazze 1013 Women's Thong High Rise Sexy Solid G-String Thong Bikini T-String Sexy Lingerie Hipsters Panties Briefs (0.4214149595505697) |
| 4 | The Blazze 0017 Men's Round Neck Half Sleeve Cotton T-Shirt for Men (0.406984000066293) | The Blazze 0017 Men's Round Neck Half Sleeve Cotton T-Shirt for Men (0.406984000066293) |
| 5 | THE BLAZZE Men's Sleeveless T-Shirt Tank Top Gym Tank Stringer Vest for Men (0.39902506271042604) | THE BLAZZE Men's Sleeveless T-Shirt Tank Top Gym Tank Stringer Vest for Men (0.39902506271042604) |
| 6 | THE BLAZZE Sleeveless T Shirts for Men (0.39881523250897205) | THE BLAZZE Sleeveless T Shirts for Men (0.39881523250897205) |
| 7 | The Blazze Navy Outdoor Seamless Bandanas Tube Headband for Men and Women (0.3889042525911601) | The Blazze Navy Outdoor Seamless Bandanas Tube Headband for Men and Women (0.3889042525911601) |
| 8 | The Blazze Outdoor Seamless Bandanas Tube,Womens and Mens Headband Headwear Headwrap (0.38213529578297223) | The Blazze Outdoor Seamless Bandanas Tube,Womens and Mens Headband Headwear Headwrap (0.38213529578297223) |
| 9 | The Blazze 0005 Men's Gym Tank Gym Stringer Gym Tank Stringer Bodybuilding Tank Tops Gym Vest Muscle Tee for Men (0.3751788632826174) | The Blazze 0005 Men's Gym Tank Gym Stringer Gym Tank Stringer Bodybuilding Tank Tops Gym Vest Muscle Tee for Men (0.3751788632826174) |
| 10 | THE BLAZZE Men's Bodybuilding Gym Solid Color Tank Top Stringers (0.37231256958123443) | THE BLAZZE Men's Bodybuilding Gym Solid Color Tank Top Stringers (0.37231256958123443) |

Ranking changed: **False**.

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

- **Fusion vs text-only:** See paired bootstrap CIs in `paired_ndcg_fusion_minus_minilm` and the generated test table.
- **Percentile vs tail-log:** Percentile calibration underperformed tail-log on validation pooled NDCG (`transform_selection_reason`, `validation_transform_ndcg10_pooled`).
- **Variant selection:** **Binary** category mode had **higher validation pooled NDCG** than affinity variant **A** and **better guardrail agreement shares** on validation (`selection_meta.validation_means`, `variant_guardrails_binary_vs_A`). Variant A was evaluated, not omitted for lack of code—the simpler binary profile won on validation and safety checks. (Ignore the legacy phrase in `selection_reason`; use the means above.)
- **Tuned weights:** Coordinate ascent proposed a candidate vector, but **guardrails rejected** it (`guardrails.accepted_tuned`, `rejection_reason`). **Hand-set defaults** in `artifacts/tuned_weights.json` ship; see `weight_search.unchanged_reason`.
- **Blazze-type queries:** Long, attribute-heavy titles stress text and imputation; see `cases.blazze` and `cases.blazze_weights_compare` ([blazze-ranking.md](blazze-ranking.md)).

## Comparability with earlier runs

Metrics in `docs/eval.json` are **not directly comparable** to earlier smoke or pre-dedup runs: co-purchase label construction, title-group splits, and dedup rules evolved (`split.hash`, `dedup_merge`). Compare only runs that share the same split hash and pipeline version.

## Latency (exact path)

Stage timings are in `run.runtime_stages_sec`. **Hotspots** in the capped run: pooled **train/eval pool builds**, **variant guardrail share audits**, and **exact validation** over 1.5k queries dominate wall time; bootstrap is cheap. Online **`latency_exact`** (500-query sample) is in eval JSON. In-process LRU caching helps repeat `/find_similar_products` calls.

**Next step (serving):** replace linear `uniq_id` dict lookup with **`numpy` structured arrays or `pandas` index** for hot paths if profiling shows Python dict overhead; keep fusion math unchanged.

**Next step (tuning):** **constrained search**—e.g. category weight floor and higher image weight cap when coverage is high—so guardrails pass without collapsing child-category agreement.

In-process LRU caching backs repeated `/find_similar_products` calls.

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
