from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from service import api as api_module
from service.config import Settings
from service.scoring import FusionWeights
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


def test_api_404(client: TestClient) -> None:
    r = client.get(
        "/find_similar_products",
        params={"product_id": "missing", "num_similar": 5},
    )
    assert r.status_code == 404


def test_api_422(client: TestClient) -> None:
    r = client.get(
        "/find_similar_products",
        params={"product_id": "id00", "num_similar": 0},
    )
    assert r.status_code == 422


def test_find_similar_returns_string_list(client: TestClient) -> None:
    r = client.get(
        "/find_similar_products",
        params={"product_id": "id00", "num_similar": 5},
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert all(isinstance(x, str) for x in data)


def test_products_search(client: TestClient) -> None:
    r = client.get("/products/search", params={"q": "Product 1", "limit": 3})
    assert r.status_code == 200
    items = r.json()
    assert len(items) <= 3
    assert len(items) >= 1
    assert "id" in items[0] and "name" in items[0]


def test_products_search_null_metadata(client: TestClient) -> None:
    cat = make_synthetic_catalog()
    cat.brands[0] = None  # type: ignore[misc]
    cat.categories[0] = None  # type: ignore[misc]
    api_module._state["catalog"] = cat
    r = client.get("/products/search", params={"q": "Product 0", "limit": 1})
    assert r.status_code == 200
    assert r.json()[0]["brand"] == ""


def test_products_search_limit_422(client: TestClient) -> None:
    r = client.get("/products/search", params={"q": "Product", "limit": 26})
    assert r.status_code == 422


def test_product_by_id(client: TestClient) -> None:
    r = client.get("/products/id00")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "id00"
    assert body["name"] == "Product 0"
    assert body["child_category"] == "Cat0"


def test_product_by_id_404(client: TestClient) -> None:
    r = client.get("/products/missing")
    assert r.status_code == 404
    body = r.json()
    assert body["detail"]
    assert body["code"] == "not_found"


def test_products_batch_order(client: TestClient) -> None:
    r = client.get("/products", params={"ids": "id02,id00,id01"})
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()]
    assert ids == ["id02", "id00", "id01"]


def test_products_batch_too_many_ids(client: TestClient) -> None:
    ids = ",".join(f"id{i:02d}" for i in range(51))
    r = client.get("/products", params={"ids": ids})
    assert r.status_code == 422


def test_demo_index(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Product Similarity Demo" in r.text


def test_meta_includes_currency_symbol(client: TestClient) -> None:
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["currency_symbol"] == "₹"
