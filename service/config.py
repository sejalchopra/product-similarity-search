"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from service.transforms import TextImageTransform

# Same query as README / demo UI (Monte Carlo men's tracksuit).
DEMO_QUERY_PRODUCT_ID = "ad8a5a196d515ef09dfdaf082bdc37c4"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    artifacts_dir: Path = Field(default=Path("artifacts"))
    search_mode: str = Field(default="exact", validation_alias="SEARCH_MODE")
    ann_pool_per_source: int = Field(default=500)
    ann_text_pool: int = Field(default=500)
    num_similar_max: int = Field(default=50)
    text_image_transform: TextImageTransform = Field(default=TextImageTransform.TAIL_LOG)
    cache_maxsize: int = Field(default=4096)
    currency_symbol: str = Field(default="₹", validation_alias="CURRENCY_SYMBOL")

    vector_backend: str = Field(default="memory")

    @property
    def is_ann(self) -> bool:
        return self.search_mode.lower() == "ann"


def load_settings() -> Settings:
    return Settings()
