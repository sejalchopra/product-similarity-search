from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service import api as api_module
from service.config import Settings
from service.scoring import BLOCK_NAMES, FusionWeights
from tests.conftest import make_synthetic_catalog


@asynccontextmanager
async def _test_lifespan(_app):
    api_module._state["catalog"] = make_synthetic_catalog()
    api_module._state["settings"] = Settings()
    api_module._state["ready"] = True
    api_module._state["weights"] = FusionWeights(
        w={
            "text": 0.30,
            "image": 0.15,
            "category": 0.15,
            "audience": 0.08,
            "brand": 0.10,
            "color": 0.08,
            "price": 0.08,
            "rating": 0.04,
            "weight": 0.02,
        },
        beta=1.0,
    )
    yield


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api_module.app.router, "lifespan_context", _test_lifespan)
    with TestClient(api_module.app) as c:
        yield c


def test_explore_contributions_sum_to_score(client: TestClient) -> None:
    r = client.get("/explore", params={"product_id": "id00", "n": 10})
    assert r.status_code == 200
    body = r.json()
    for item in body["results"]:
        total = sum(item["blocks"][b]["contribution"] for b in BLOCK_NAMES)
        assert abs(total - item["score"]) < 1e-6


def test_explore_matches_find_similar(client: TestClient) -> None:
    sim = client.get(
        "/find_similar_products",
        params={"product_id": "id00", "num_similar": 5},
    ).json()
    exp = client.get("/explore", params={"product_id": "id00", "n": 5}).json()
    assert [x["id"] for x in exp["results"]] == sim


def test_explore_unknown_404(client: TestClient) -> None:
    r = client.get("/explore", params={"product_id": "missing", "n": 5})
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


def test_explore_n_over_200_422(client: TestClient) -> None:
    r = client.get("/explore", params={"product_id": "id00", "n": 201})
    assert r.status_code == 422


def test_explore_mask_brand_imputed(client: TestClient) -> None:
    r = client.get(
        "/explore",
        params={"product_id": "id01", "n": 5, "mask": "brand"},
    )
    assert r.status_code == 200
    reasons = {
        item["blocks"]["brand"]["imputed_reason"]
        for item in r.json()["results"]
        if item["blocks"]["brand"]["imputed"]
    }
    assert "query brand missing" in reasons


def test_eval_endpoint(client: TestClient) -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "eval.json"
    if not path.is_file():
        pytest.skip("eval.json missing")
    r = client.get("/eval")
    assert r.status_code == 200
    data = r.json()
    import json

    assert data["row_count"] == json.loads(path.read_text())["row_count"]


def test_explore_currency_and_category_key(client: TestClient) -> None:
    cat = make_synthetic_catalog()
    cat.categories[0] = "MensT_Shirts"
    api_module._state["catalog"] = cat
    r = client.get("/explore", params={"product_id": "id00", "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["currency_symbol"] == "₹"
    assert body["query"]["child_category_key"] == "MensT_Shirts"
    assert body["query"]["child_category"] == "Men's T-Shirts"
