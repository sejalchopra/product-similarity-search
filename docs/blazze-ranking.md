# Blazze G-string ranking

Query `12d33c127417b634899aa4c7cbe747f1`: "The Blazze Men's Modal V Back G String Sexy Low Rise Briefs Panties, Men Boxer Underpants Shorts Underwear Bulge Pouch Funny Thongs".

Child category `MensThongUnderwear`, audience men, brand `the blazze`, price 199. The query has no image, no color, `rating_n` 0, and no weight. Exact mode, shipped weights (`tail_log`). No code was changed.

## Counts

- Same child category: **2** products, including the query. The only other one is La Intimo men's lace thong (`8bb771ee90c2ad9ae7c342aca9b9d4cb`).
- Name contains "thong" or "brief" (case-insensitive) and audience is men: **520**, including the query. Of those, 9 contain "thong" and 519 contain "brief" (the query title contains both). Most of the 520 are filed as `MensUnderwearBriefs` (484), not `MensThongUnderwear` (2).

There is no set of five same-category products outside the top 20. The other `MensThongUnderwear` row is rank 2.

## Top 10

Weights: text 0.30, image 0.15, category 0.15, audience 0.08, brand 0.10, color 0.08, price 0.08, rating 0.04, weight 0.02.

On every row below, image, color, rating, and weight are imputed (value 0.50; contributions 0.075, 0.040, 0.020, 0.010). Those four blocks add 0.145 to every candidate and do not change the order. The other five blocks are observed. Cells are value / contribution.

| Rank | Product | Category | Score | Text | Category | Audience | Brand | Price |
|------|---------|----------|------:|-----:|---------:|---------:|------:|------:|
| 1 | Blazze modal G-string, no "V Back" | MensUnderwearBriefs | 0.583 | 0.658 / 0.198 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.753 / 0.060 |
| 2 | La Intimo lace thong | MensThongUnderwear | 0.510 | 0.358 / 0.107 | 1 / 0.150 | 1 / 0.080 | 0 / 0 | 0.352 / 0.028 |
| 3 | Blazze women's thong | WomensG_StringThongPanties | 0.476 | 0.505 / 0.151 | 0 / 0 | 0 / 0 | 1 / 0.100 | 0.996 / 0.080 |
| 4 | Blazze cotton T-shirt | MensT_Shirts | 0.476 | 0.236 / 0.071 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.996 / 0.080 |
| 5 | Blazze sleeveless T-shirt | MensT_Shirts | 0.468 | 0.261 / 0.078 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.808 / 0.065 |
| 6 | Blazze stringer vest | MensSportsT_ShirtsJerseys | 0.467 | 0.209 / 0.063 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.996 / 0.080 |
| 7 | Blazze bandana | MensCapsHats | 0.439 | 0.113 / 0.034 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.996 / 0.080 |
| 8 | Blazze tank stringer | MensT_Shirts | 0.438 | 0.162 / 0.049 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.808 / 0.065 |
| 9 | Blazze gym tank (category says briefs) | MensUnderwearBriefs | 0.435 | 0.190 / 0.057 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.670 / 0.054 |
| 10 | Blazze boy shorts | SportsFitnessOutdoors | 0.427 | 0.209 / 0.063 | 0 / 0 | 0.5 / 0.040 | 1 / 0.100 | 0.996 / 0.080 |

Rank 3 audience is an observed women-vs-men mismatch, not an imputed value. Rank 10 audience is the unisex table value.

## Same-category products outside the top 20

None. The best men's "thong"/"brief" titles that missed the top 20 are a different child category (`MensUnderwearBriefs`). Same imputed blocks as above.

| Product | Score | Text | Category | Audience | Brand | Price |
|---------|------:|-----:|---------:|---------:|------:|------:|
| MYZA camouflage g-string thong | 0.390 | 0.333 / 0.100 | 0 / 0 | 1 / 0.080 | 0 / 0 | 0.808 / 0.065 |
| Blazze brief, pack of 7 (price 1399) | 0.383 | 0.176 / 0.053 | 0 / 0 | 1 / 0.080 | 1 / 0.100 | 0.069 / 0.006 |
| BODYCARE cotton briefs | 0.380 | 0.260 / 0.078 | 0 / 0 | 1 / 0.080 | 0 / 0 | 0.956 / 0.077 |
| Semantic cotton briefs | 0.378 | 0.299 / 0.090 | 0 / 0 | 1 / 0.080 | 0 / 0 | 0.791 / 0.063 |
| BASIICS feather-weight brief | 0.371 | 0.271 / 0.081 | 0 / 0 | 1 / 0.080 | 0 / 0 | 0.808 / 0.065 |

## Why T-shirts outrank other underwear

They do not outrank the same child category. La Intimo is the only other `MensThongUnderwear` product and it is rank 2, ahead of every T-shirt. Its category contribution is 0.150. A same-brand T-shirt's brand contribution is 0.100. Category wins that comparison.

T-shirts outrank other men's briefs. Those briefs are `MensUnderwearBriefs`, so the category block is 0 for the T-shirt and for the brief. The 0.15 category weight never treats "men's underwear" as one family.

Against the strongest missed brief (MYZA, 0.390) the rank-4 T-shirt (0.476) is ahead by 0.086:

- Brand: +0.100 for the T-shirt, 0 for MYZA.
- Text: the T-shirt is worse by 0.029 contribution (0.071 vs 0.100). "Stringer" tanks still get a mid text score because the embedding sits near "G String", and the titles share "The Blazze" and "Men".
- Price: +0.015. The query costs 199, and several T-shirts cost 199. The Blazze 7-pack at 1399 drops price contribution to 0.006, which is why that same-brand brief loses.
- Image, color, rating, and weight: no difference. The query is missing all four, so every candidate is imputed to 0.50.

Rank 1 is the other Blazze G-string (raw text cosine 0.981). It is not the same child category, so it gets no category credit. Text plus brand is enough.

## Tuned weights and same-category rate

Shipped weights are in `artifacts/tuned_weights.json`: text 0.30, image 0.15, category 0.15, audience 0.08, brand 0.10, color 0.08, price 0.08, rating 0.04, weight 0.02. Transform `tail_log`. Weight search did not replace them; see `guardrails.accepted_tuned` in `docs/eval.json`.

The current catalog has 21,627 title groups (`title_group_hist.unique_groups`). Same-child share on the capped test split is `guardrails_test.shipped.same_child_share10` in that file. This query is the sparse case: `MensThongUnderwear` has one other product, so fusion can place only one same-category item in the top 10, and it does.
