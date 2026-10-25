# Product Similarity Search

Given a product ID, returns the most similar products from a 30,000-item Amazon
fashion catalog. The catalog is too messy for a single similarity formula, so each
attribute is scored separately, calibrated to a common scale, and combined with
weights. Every result can be explained, and the demo UI shows why a product ranked
where it did.

## Quick start

**Local**

Needs Python 3.11 or newer. On this Mac, `python3` is 3.9, so the venv below
uses `python3.12`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make run-local          # http://localhost:8000
```

**Docker**

```bash
make build
docker run --rm -p 8000:8000 product-similarity:local
```

**Kubernetes** (tested on a local kind cluster)

```bash
make build
kind load docker-image product-similarity:local
kubectl apply -f k8s.yaml
kubectl port-forward svc/product-similarity 8000:80
```

The Service is named `product-similarity`. It listens on port 80 and forwards to
container port 8000. `imagePullPolicy` is `IfNotPresent`, so load the image into
the cluster before the pod can start.

Embeddings are already in `artifacts/`, so startup does not download models. The
catalog is already unzipped at
`data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`
(about 71 MB). `data/archive.zip` is not in git. If that ldjson is missing and you
have the zip, run `cd data && unzip archive.zip`. `make index` rebuilds the
embeddings and needs `requirements-index.txt` (PyTorch). Run that only if the
catalog changes.

## API

| Endpoint | Purpose |
|---|---|
| `GET /find_similar_products?product_id=...&num_similar=...` | Required endpoint. JSON list of product IDs. 404 for an unknown product, 422 if `num_similar` is outside 1–50. The list can be shorter than requested after de-duplication. |
| `GET /explore?product_id=...&n=100` | Top results with the per-block score breakdown (used by the demo). Optional `mask=brand` or `mask=image`. |
| `GET /explain?product_id=...&neighbor_id=...` | Score breakdown for one product pair. |
| `GET /eval`, `GET /meta` | Evaluation report and the active configuration. |
| `GET /health`, `GET /ready` | Liveness and readiness (used by `k8s.yaml`). |
| `GET /docs` | Interactive OpenAPI documentation. |

```bash
curl "http://localhost:8000/find_similar_products?product_id=ad8a5a196d515ef09dfdaf082bdc37c4&num_similar=5"
curl "http://localhost:8000/find_similar_products?product_id=52e31bb31680b0ec73de0d781a23cc0a&num_similar=10"
```

The first id is a men's Monte Carlo tracksuit; the five hits are other men's
tracksuits. The second is a women's kurti; the hits are other kurtis.

## Demo UI

Open `http://localhost:8000`. Search for "Monte Carlo Grey Solid Cotton Blend
Polo Collar Tracksuit" (its brand field is empty). Open a result to see the score
split by attribute, drag the weight sliders to re-rank the fetched list, and use
**Hide brand** or **Hide image** to force those blocks onto the missing-value
score. Longer notes: [docs/DESIGN.md](docs/DESIGN.md).

## How it works

- **Nine attribute blocks:** title text (MiniLM embeddings), image (CLIP embeddings), category, audience, brand, color, price, rating, weight. Each score is in [0, 1]; the final score is their weighted average. Weight is rarely present on both products, so it seldom changes the order.
- **Calibration:** text and image raw cosines are not on the same scale as the other blocks. They use a tail-log of their percentile among random pairs, which keeps the best matches spread out. Tail-log beat raw and percentile on validation; both paired intervals exclude zero.
- **Missing data:** a missing value gets the score a random pair would get. Brand, category, and audience imputations are capped at 0.5, so a gap cannot look like an exact match.
- **De-duplication:** listings that differ only in color or size share a title group. The query's group is excluded, and each other group contributes at most one product.
- **Exact vs approximate:** exact scoring of the full catalog is the default. With `SEARCH_MODE=ann`, candidates come from an HNSW text index plus same-brand and same-category lists, then the same scorer re-ranks them. Reference: Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 42(9), 2020 (arXiv:1603.09320).
- **Offline vs online:** embeddings are computed once offline. A request does arithmetic only, and the image installs `requirements.txt` (the serving stack).

Detail and the generated metric tables: [docs/DESIGN.md](docs/DESIGN.md).

## Evaluation

There are no human similarity labels. Relevance is the catalog's "customers also
bought" field, matched by normalized title (noisy, popularity-skewed). Queries are
split by title group into train, validation, and test. Popularity and text-only
(MiniLM) baselines are compared with paired cluster-bootstrap confidence
intervals. The shipped report is a **capped** run (1,500 tune, 1,500 validation,
2,000 test queries).

Headline: fusion beats text-only on NDCG@10, and the paired 95% confidence
interval on the difference excludes zero on both validation and test.

Tables are generated into [docs/DESIGN.md](docs/DESIGN.md). The report is
[docs/eval.json](docs/eval.json) (`paired_ndcg_fusion_minus_minilm`).

## Key decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Separate blocks instead of one vector | Each attribute is a different kind of mess, and a result can be explained | More parts to calibrate |
| Hand-set weights shipped | Weight search on the noisy label cut same-category agreement, so a guardrail rejected it | Some label-measured quality stays unused |
| Tail-log calibration | Beat raw and percentile on validation | Needs a stored sample of random pairs |
| Embeddings precomputed offline | Requests stay arithmetic and the image stays small | Rebuild if the catalog changes |
| In-process LRU cache | The catalog is static | Replicas do not share the cache |
| Exact search by default | The ranking is the true fused order, and this catalog is small enough | A much larger catalog needs `SEARCH_MODE=ann` |

## Limitations and next steps

- The labels are noisy. Read the metrics as comparisons between methods.
- Some queries rank a neighboring product type above a closer one when the category is sparse. See the Blazze G-string case in [docs/blazze-ranking.md](docs/blazze-ranking.md) and the generated before/after table in [docs/DESIGN.md](docs/DESIGN.md). A category-affinity variant was tested; the binary category score won on validation.
- Uncached exact queries are about 170 ms (p50 in `docs/eval.json` → `latency_exact`). The slowest blocks in that profile are price, image, and text.
- `docs/bench_hnsw.json` is a 5,000-vector smoke run of pure HNSW. `make bench` defaults to 1,000,000 synthetic vectors. Neither measures the full candidate union (HNSW plus brand and category lists).
- Next: faster block lookups, weight search with a floor on category agreement, and brand inference from titles (`BRAND_BACKFILL`, off by default).

## Repository layout

`app.py` (uvicorn entry) · `service/` (API, scoring, catalog) · `scripts/` (index build, benchmark) · `scripts/eval/` (evaluation pipeline) · `artifacts/` (precomputed embeddings and manifest) · `docs/` (design note, eval report, Blazze case) · `tests/` · `k8s.yaml` · `Dockerfile`

## Make targets

`run-local` · `build` · `index` · `eval-smoke` · `eval-medium` · `eval` · `bench` · `lint` · `test` · `demo-warm`

---

# Original assignment

# sap-cxii-tech-ex-01
Technical Exercise for SAP. 

## What is the goal? 
You are tasked with implementing a product similarity search feature using the given dataset. Given a product’s `unique ID` (uniq_id), your goal is to retrieve a list of `similar products` based on `specific attributes`. As you think and solve the problem below, 

### Explain your design decisions and trade-offs
### Suggested Approaches:
- Implement approximate nearest neighbors using libraries like FAISS, Annoy, or HNSW
- Utilize dimensionality reduction techniques (e.g., PCA, t-SNE) before similarity computation
- Consider caching strategies for frequently accessed products

### Multimodal Similarity Search (Optional)
- Extend the similarity search to incorporate both text and image features, creating a multimodal product recommendation system.
#### Text-Based Similarity:
- Extract meaningful features from product descriptions, names, and other text attributes-
- Use natural language processing techniques to capture semantic similarity between products
- Implement techniques like TF-IDF, word embeddings, or transformer-based embeddings
  
#### Image-Based Similarity:
- For products with available image URLs, download and process the images
- Extract visual features using pre-trained convolutional neural networks (e.g., ResNet, EfficientNet)
- Calculate visual similarity between products

### Combined Approach:
- Develop a method to combine text-based and image-based similarity scores
- Allow configurable weighting between different similarity components
- Implement a fallback mechanism when certain features are unavailable.

--- 

## Part 1: Similarity Search Function

Write a function called `find_similar_products` that takes in the following parameters:

```python
def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    pass
```

The function should return a list of num_similar product IDs that are most similar to the given product_id. 
The similarity between products should be determined based on the following attributes: `brand, color, price, image, sales_price, weight, and rating` NOTE You can dedcide if you also want to use other attributes in the dataset.
You should consider the following requirements while implementing the function:
 - The function should first retrieve the attributes `(brand, sales_price, weight, rating, .. all others you pick)` of the product identified by the given `product_id`.
 - The function should then compare these attributes with all other products in the dataset.
 - Calculate a similarity score between the given product and each other product using a suitable similarity measure `(e.g., Euclidean distance, cosine similarity, etc.)`.
 - Sort the products based on their similarity scores in descending order.
 - Return a `list of num_similar product IDs` that are most similar to the given `product_id`.
   
### Here is the sample code

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Assume df is already loaded with the dataset
## You could write a better data loader for this as well, and find an alternative to pandas.
df = pd.read_json('data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson', lines=True)

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    pass

## This is a sample stub you can chose the datastructures you deem important.
def calculate_similarity():
    pass
```

## Note:
  - If multiple products have the same similarity score, the tie-breaker can be based on any attribute (e.g., sales_price or rating).

- ### Assumptions: 
  - You can assume that the given `product_id` will exist in the dataset.
  - You can use the provided DataFrame df to access the dataset.
  - Please write the code for the `find_similar_products` function and any additional helper functions you might need, as included in the sample above.

## Part 2: Microservice Implementation
- `Required`: Write a microservice with an endpoint `GET /find_similar_products` which takes two request params(`product_id`, `num_similar`) and returns the result. Use FastAPI for the web app.
  - You can read about [FastAPI](https://fastapi.tiangolo.com/)
- `Required`: Make the image to be deployable on kubernetes with the sample given for the `Dockerfile`
  - You can read about [Docker](https://docs.docker.com/)
  - `Optional`: You don't have to do it on, but if you have experience you can deploy on your local cluster however you want. [Kubernetes](https://kubernetes.io/)
  
### Sample Code for FastAPI: 
```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.get("/find_similar_products")
def get_similar_products(product_id: str, num_similar: int) -> List[str]:
    try:
        ## You can implement it differently.
        similar_products = find_similar_products(product_id, num_similar)
        return similar_products
    except Exception as e:
        raise e ## Enhance this error case with different HTTP Error Codes.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Bonus - Part 3: Optimization Problem with Vector Searches (Bonus)
- You are tasked with optimizing the similarity search function to handle large datasets efficiently using vector searches. 
- Implement this using any similarity search alogrithm avaiable on hugging face or even OSS papers for faster nearest neighbor search.
- You should reference the paper and algorithms you use, and you should be able to explain why you chose this implementation.


## Extra How to Help? 
## Amazon Dataset for Apparel: 
- You will be working with the `2020` Amazon data below: 
Ref: [Kaggle](https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020)
- Download and unzip the file on local (if you don't have a kaggle account we have checked in the file under `data`) 
- Unzip and you should have a file in `ldjson` format.
- For data exploration look at all the attributes that are avaialbe to you. 
  - `cd data; unzip archive.zip` 
```bash
.
├── archive.zip
└── marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson

1 directory, 2 files
```
