from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore", env_parse_none_str="none"
    )

    # --- providers (all optional: the pipeline degrades gracefully) ---
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"
    typesafe_api_key: str | None = None  # Jev
    typesafe_model: str = "jev-latest"

    # --- repeatability: same input -> same FSC codes ---
    gemini_seed: int | None = 7  # fixed sampling seed on every Gemini call (None = random each call)
    gemini_temperature: float | None = None  # None = model default for gemini-3*, per-call value for older models
    llm_cache: bool = True  # memoise Gemini outputs on disk, keyed by model + prompt + inputs (see app/llm/cache.py)

    # auto = jev if key, else gemini if key, else retrieval-only
    decider: Literal["auto", "jev", "gemini", "retrieval"] = "auto"
    enable_web_research: bool = True  # Gemini + Google Search grounding when evidence is sparse

    # --- data / cache ---
    fsc_xlsx_path: Path = REPO_DIR / "data" / "FSC_DLA_GSA_assignments.xlsx"
    cache_dir: Path = BACKEND_DIR / ".cache"

    # --- http ---
    cors_origins: list[str] = ["http://localhost:3001", "http://127.0.0.1:3001"]
    max_upload_mb: int = 20
    max_files: int = 10

    # --- ingest ---
    max_site_pages: int = 6
    max_evidence_chars: int = 120_000
    sparse_evidence_chars: int = 1_500  # below this we trigger web research

    # --- retrieval / decision ---
    k_per_query: int = 12
    k_total: int = 60
    jev_batch_size: int = 40  # noul questions per Jev request
    jev_weight: float = 0.75  # final = w * decider_prob + (1-w) * retrieval_norm

    # --- output selection ---
    min_score: float = 0.45
    max_results: int = 10
    min_results: int = 3  # always return at least this many (flagged low confidence)


@lru_cache
def get_settings() -> Settings:
    return Settings()
