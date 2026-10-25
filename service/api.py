"""FastAPI application."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, Response

from service.catalog import Catalog, load_catalog
from service.config import Settings, load_settings
from service.scoring import FusionWeights
from service.similarity import explain_one, find_similar_products
from service.ui_api import router as ui_router

logger = logging.getLogger("product_similarity")
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)

_state: dict[str, object] = {}


class ErrorBody(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
    code: str = Field(
        ...,
        description="Machine code",
        examples=["not_found"],
    )


class ExplainBlock(BaseModel):
    value: float = Field(..., description="Calibrated block contribution")
    raw: float | None = Field(
        default=None,
        description="Raw similarity before calibration",
    )
    imputed: bool = Field(..., description="True if this block was imputed")


class ExplainResponse(BaseModel):
    product_id: str = Field(..., examples=["26d41bdc1495de290bc8e6062d927729"])
    neighbor_id: str = Field(..., description="Compared product id")
    product_name: str = Field(..., description="Neighbor display name")
    score: float = Field(..., description="Fused similarity score")
    transform: str = Field(..., description="Text/image transform in use")
    blocks: dict[str, ExplainBlock]


class SimilarQuery(BaseModel):
    product_id: str = Field(
        ...,
        description="Catalog uniq_id",
        examples=["26d41bdc1495de290bc8e6062d927729"],
    )
    num_similar: int = Field(
        ...,
        ge=1,
        le=50,
        description="Number of similar products to return (may return fewer after dedup)",
        examples=[5],
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    if not (settings.artifacts_dir / "manifest.json").exists():
        logger.warning("artifacts missing; /ready will be false")
        _state["catalog"] = None
        _state["settings"] = settings
        _state["ready"] = False
        yield
        return
    from service.artifacts_validate import validate_tuned_weights_file

    catalog = load_catalog(settings.artifacts_dir, build_hnsw=settings.is_ann)
    validate_tuned_weights_file(settings.artifacts_dir)
    tuned_path = settings.artifacts_dir / "tuned_weights.json"
    tuned_source = "defaults"
    tuned_meta: dict[str, object] = {}
    if tuned_path.exists():
        tw = json.loads(tuned_path.read_text())
        tuned_source = str(tw.get("source", "full"))
        git_field = tw.get("git")
        git_sha = git_field.get("sha") if isinstance(git_field, dict) else tw.get("git_sha")
        tuned_meta = {
            "path": str(tuned_path.name),
            "source": tuned_source,
            "timestamp_utc": tw.get("timestamp_utc"),
            "git_sha": git_sha,
            "split": tw.get("split"),
        }
        logger.info(
            "loaded tuned weights from %s source=%s variant=%s transform=%s",
            tuned_path.name,
            tuned_source,
            tw.get("selected_variant"),
            tw.get("text_image_transform"),
        )
        from service.scoring_profile import ScoringProfile
        from service.transforms import TextImageTransform

        variant = str(
            tw.get("selected_variant", catalog.manifest.get("selected_variant", "binary"))
        )
        catalog.active_profile = ScoringProfile.from_variant(variant)
        catalog.manifest["selected_variant"] = variant
        _state["weights"] = FusionWeights(
            w=tw["w"],
            beta=float(tw.get("beta", 1.0)),
            text_image_transform=TextImageTransform(tw.get("text_image_transform", "tail_log")),
        )
    _state["catalog"] = catalog
    _state["settings"] = settings
    _state["tuned_weights_meta"] = tuned_meta
    _state["ready"] = True
    if not tuned_path.exists():
        _state["weights"] = FusionWeights(
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
            text_image_transform=settings.text_image_transform,
        )
    yield


app = FastAPI(
    title="Product Similarity",
    description=(
        "Hybrid late-fusion similarity over the Amazon fashion catalog. "
        "Results may contain fewer than num_similar ids when title-group dedup "
        "exhausts distinct products."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "path=%s status=%s latency_ms=%.2f",
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _catalog() -> Catalog:
    cat = _state.get("catalog")
    if cat is None:
        raise HTTPException(
            status_code=503,
            detail="Artifacts not loaded",
        )
    return cat  # type: ignore[return-value]


def _settings() -> Settings:
    return _state["settings"]  # type: ignore[return-value]


@lru_cache(maxsize=4096)
def _cached_top50(product_id: str, mode: str) -> tuple[str, ...]:
    catalog = _catalog()
    settings = _settings()
    weights = _state.get("weights")
    return tuple(
        find_similar_products(
            catalog,
            product_id,
            50,
            settings,
            weights=weights,  # type: ignore[arg-type]
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": bool(_state.get("ready"))}


@app.get(
    "/find_similar_products",
    response_model=list[str],
    responses={
        404: {"model": ErrorBody},
        422: {"model": ErrorBody},
        503: {"model": ErrorBody},
    },
)
def get_similar_products(
    product_id: Annotated[
        str,
        Query(description="Product uniq_id", examples=["26d41bdc1495de290bc8e6062d927729"]),
    ],
    num_similar: Annotated[int, Query(ge=1, le=50, description="Max results", examples=[5])],
) -> list[str]:
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Not ready", headers={})
    catalog = _catalog()
    settings = _settings()
    if product_id not in catalog.id_to_row:
        raise HTTPException(status_code=404, detail="Unknown product_id")
    mode = "ann" if settings.is_ann else "exact"
    return list(_cached_top50(product_id, mode)[:num_similar])


@app.get("/explain", response_model=ExplainResponse)
def explain(
    product_id: Annotated[str, Query(description="Query uniq_id")],
    neighbor_id: Annotated[str, Query(description="Neighbor uniq_id")],
) -> ExplainResponse:
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Not ready")
    catalog = _catalog()
    settings = _settings()
    if product_id not in catalog.id_to_row or neighbor_id not in catalog.id_to_row:
        raise HTTPException(status_code=404, detail="Unknown id")
    weights = _state.get("weights")
    if not isinstance(weights, FusionWeights):
        raise HTTPException(status_code=503, detail="Not ready")
    raw: dict[str, Any] = explain_one(
        catalog,
        product_id,
        neighbor_id,
        settings,
        weights=weights,
    )
    blocks_dict = raw["blocks"]
    raw_map = raw["raw"]
    imputed_map = raw["imputed"]
    blocks = {
        str(k): ExplainBlock(
            value=float(blocks_dict[k]),
            raw=raw_map.get(k) if isinstance(raw_map, dict) else None,
            imputed=bool(imputed_map.get(k, False)) if isinstance(imputed_map, dict) else False,
        )
        for k in blocks_dict
    }
    return ExplainResponse(
        product_id=product_id,
        neighbor_id=neighbor_id,
        product_name=str(raw["product_name"]),
        score=float(raw["score"]),
        transform=str(raw["transform"]),
        blocks=blocks,
    )


app.include_router(ui_router)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
