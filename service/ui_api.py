"""Read-only product and meta endpoints for the demo UI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from service.catalog import Catalog
from service.cleaning import humanize_category_key
from service.config import Settings
from service.explore import explore_similar
from service.query_overrides import QueryOverrides
from service.scoring import FusionWeights

router = APIRouter(tags=["products"])


def _catalog() -> Catalog:
    from service.api import _catalog as get_catalog

    return get_catalog()


def _settings() -> Settings:
    from service.api import _settings as get_settings

    return get_settings()


def _state() -> dict[str, object]:
    from service.api import _state as state

    return state


class ProductSummary(BaseModel):
    id: str = Field(..., description="Catalog uniq_id", examples=["id00"])
    name: str = Field(..., description="Product display name", examples=["Product 0"])
    brand: str = Field(..., description="Brand name (may be empty)", examples=["BrandA"])
    price: float | None = Field(
        None,
        description="Sales price in catalog currency",
        examples=[100.0],
    )
    image_url: str | None = Field(
        None,
        description="First medium image URL, or null if none",
        examples=["https://example.com/img.jpg"],
    )


class ProductCard(ProductSummary):
    rating: float | None = Field(None, description="Average rating", examples=[4.0])
    child_category: str = Field(
        ...,
        description="Humanized child category label",
        examples=["Cat0"],
    )
    audience: str = Field(..., description="Audience label", examples=["women"])


class ExploreBlock(BaseModel):
    value: float
    raw_sim: float | None = None
    weight: float
    contribution: float
    imputed: bool
    imputed_reason: str | None = None


class ExploreResult(ProductCard):
    child_category_key: str = ""
    weight_g: float | None = None
    color_missing: bool = False
    score: float
    rank: int
    blocks: dict[str, ExploreBlock]


class ExploreQuery(ProductCard):
    child_category_key: str = ""
    weight_g: float | None = None
    color_missing: bool = False


class ExploreMeta(BaseModel):
    candidates_scored: int
    title_groups_total: int
    hidden_query_group_count: int
    hidden_query_group_ids: list[str]
    duplicates_collapsed: int
    mode: str
    transform: str
    weights: dict[str, float]
    latency_ms: float
    cache_hit: bool
    currency_symbol: str


class ExploreResponse(BaseModel):
    query: ExploreQuery
    results: list[ExploreResult]
    meta: ExploreMeta


class MetaResponse(BaseModel):
    mode: str = Field(..., description="Search mode: exact or ann", examples=["exact"])
    transform: str = Field(
        ...,
        description="Text/image transform in use",
        examples=["tail_log"],
    )
    currency_symbol: str = Field(
        ...,
        description="Symbol for catalog sales prices (no conversion)",
        examples=["₹"],
    )
    weights: dict[str, float] = Field(
        ...,
        description="Fusion block weights (normalized for display)",
        examples=[{"text": 0.3, "image": 0.15}],
    )
    image_coverage: float = Field(0.0, description="Fraction of rows with CLIP images")
    image_weight_zeroed: bool = Field(
        False,
        description="True when image coverage < 5% so tuning fixes image weight to 0",
    )
    explore_scoring_note: str = Field(
        "",
        description="How GET /explore scores candidates",
    )
    tuned_weights_source: str = Field(
        "defaults",
        description="Provenance: defaults, smoke, or full eval",
    )
    tuned_weights_path: str | None = Field(
        None,
        description="Artifact file loaded at startup, if any",
    )


def _not_found(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": detail, "code": "not_found"},
    )


def _price_or_none(val: float) -> float | None:
    if not math.isfinite(val):
        return None
    return float(val)


def _image_url_or_none(url: str | None) -> str | None:
    if not url or not str(url).strip():
        return None
    return str(url)


def _text_or_empty(val: str | None) -> str:
    if val is None:
        return ""
    return str(val)


def _card_for_row(catalog: Catalog, row: int) -> ProductCard:
    p = float(catalog.price[row])
    r = float(catalog.rating[row])
    return ProductCard(
        id=catalog.uniq_ids[row],
        name=_text_or_empty(catalog.product_names[row]),
        brand=_text_or_empty(catalog.brands[row]),
        price=_price_or_none(p),
        image_url=_image_url_or_none(catalog.image_urls[row]),
        rating=_price_or_none(r) if int(catalog.rating_n[row]) > 0 else None,
        child_category=humanize_category_key(catalog.categories[row] or None),
        audience=_text_or_empty(catalog.audiences[row]),
    )


def _require_ready() -> None:
    if not _state().get("ready"):
        raise HTTPException(status_code=503, detail="Not ready")


@router.get("/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    _require_ready()
    settings = _settings()
    weights = _state().get("weights")
    if not isinstance(weights, FusionWeights):
        raise HTTPException(status_code=503, detail="Not ready")
    w_norm = weights.normalized().w
    mode = "ann" if settings.is_ann else "exact"
    cat = _catalog()
    cov = float(cat.manifest.get("image_coverage", 0.0) or 0.0)
    zero_img = cov < 0.05
    note = "Explore always uses exact full-catalog scoring."
    if zero_img:
        note += " Image block weight is 0 in tuning (coverage < 5%)."
    tw_meta = _state().get("tuned_weights_meta")
    tw_source = "defaults"
    tw_path: str | None = None
    if isinstance(tw_meta, dict):
        tw_source = str(tw_meta.get("source", "full"))
        raw_path = tw_meta.get("path")
        tw_path = str(raw_path) if raw_path is not None else None
    return MetaResponse(
        mode=mode,
        transform=weights.text_image_transform.value,
        currency_symbol=settings.currency_symbol,
        image_coverage=cov,
        image_weight_zeroed=zero_img,
        explore_scoring_note=note,
        weights={k: float(w_norm[k]) for k in w_norm},
        tuned_weights_source=tw_source,
        tuned_weights_path=tw_path,
    )


@router.get("/products/search", response_model=list[ProductSummary])
def search_products(
    q: Annotated[str, Query(min_length=1, description="Case-insensitive substring on name")],
    limit: Annotated[int, Query(ge=1, le=25, description="Max results")] = 10,
) -> list[ProductSummary]:
    _require_ready()
    catalog = _catalog()
    needle = q.lower()
    hits = np.flatnonzero(np.char.find(catalog.names_lower, needle) >= 0)
    out: list[ProductSummary] = []
    for row in hits[:limit]:
        i = int(row)
        card = _card_for_row(catalog, i)
        out.append(
            ProductSummary(
                id=card.id,
                name=card.name,
                brand=card.brand,
                price=card.price,
                image_url=card.image_url,
            )
        )
    return out


@router.get("/products/random", response_model=ProductCard)
def random_product() -> ProductCard:
    _require_ready()
    catalog = _catalog()
    row = int(np.random.default_rng().integers(0, catalog.n))
    return _card_for_row(catalog, row)


@router.get("/products", response_model=list[ProductCard])
def products_batch(
    ids: Annotated[
        str,
        Query(description="Comma-separated uniq_ids, max 50, order preserved"),
    ],
) -> list[ProductCard]:
    _require_ready()
    catalog = _catalog()
    parts = [p.strip() for p in ids.split(",") if p.strip()]
    if len(parts) > 50:
        raise HTTPException(status_code=422, detail="At most 50 ids allowed")
    out: list[ProductCard] = []
    for pid in parts:
        row = catalog.id_to_row.get(pid)
        if row is None:
            continue
        out.append(_card_for_row(catalog, row))
    return out


@router.get(
    "/products/{uniq_id}",
    response_model=ProductCard,
    responses={404: {"description": "Unknown product"}},
)
def product_by_id(uniq_id: str) -> ProductCard | JSONResponse:
    _require_ready()
    catalog = _catalog()
    row = catalog.id_to_row.get(uniq_id)
    if row is None:
        return _not_found("Unknown product_id")
    return _card_for_row(catalog, row)


_EVAL_PATH = Path(__file__).resolve().parents[1] / "docs" / "eval.json"


def _parse_mask(mask: str | None) -> QueryOverrides:
    if not mask or not mask.strip():
        return QueryOverrides()
    parts = {p.strip().lower() for p in mask.split(",") if p.strip()}
    allowed = {"brand", "image"}
    unknown = parts - allowed
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown mask fields: {sorted(unknown)}")
    return QueryOverrides(mask_brand="brand" in parts, mask_image="image" in parts)


_EXPLORE_CACHE: dict[tuple[int, str, int, str, str], str] = {}


@router.get("/eval")
def get_eval() -> JSONResponse:
    if not _EVAL_PATH.is_file():
        return _not_found("eval.json missing; run make eval")
    data = json.loads(_EVAL_PATH.read_text())
    return JSONResponse(content=data)


@router.get(
    "/explore",
    response_model=ExploreResponse,
    responses={404: {"description": "Unknown product"}},
)
def get_explore(
    product_id: Annotated[str, Query(description="Query uniq_id")],
    n: Annotated[int, Query(ge=1, le=200)] = 100,
    mask: Annotated[
        str | None,
        Query(description="Comma-separated: brand, image"),
    ] = None,
) -> ExploreResponse | JSONResponse:
    _require_ready()
    catalog = _catalog()
    if product_id not in catalog.id_to_row:
        return _not_found("Unknown product_id")
    settings = _settings()
    weights = _state().get("weights")
    if not isinstance(weights, FusionWeights):
        raise HTTPException(status_code=503, detail="Not ready")
    try:
        ov = _parse_mask(mask)
    except HTTPException:
        raise
    w_norm = weights.normalized().w
    weights_key = ",".join(f"{k}={w_norm[k]:.8f}" for k in sorted(w_norm))
    mask_key = mask or ""
    cache_key = (id(catalog), product_id, n, mask_key, weights_key)
    if cache_key in _EXPLORE_CACHE:
        raw: dict[str, Any] = json.loads(_EXPLORE_CACHE[cache_key])
        raw["meta"]["cache_hit"] = True
    else:
        raw = explore_similar(catalog, product_id, n, settings, weights, ov)
        raw["meta"]["cache_hit"] = False
        if len(_EXPLORE_CACHE) >= 256:
            _EXPLORE_CACHE.clear()
        _EXPLORE_CACHE[cache_key] = json.dumps(raw)
    return ExploreResponse.model_validate(raw)
