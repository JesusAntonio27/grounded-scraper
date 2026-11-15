"""Environment-only configuration. Importing the API never reads a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from anthropic import AsyncAnthropic

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; grounded-scraper/1.0)"


@dataclass(frozen=True)
class Settings:
    fetch_backend: str
    oxylabs_user: str | None
    oxylabs_pass: str | None
    oxylabs_geo_location: str | None
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    internal_token: str | None
    user_agent: str
    ocr_languages: str

    @property
    def fetch_configured(self) -> bool:
        if self.fetch_backend == "oxylabs":
            return bool(self.oxylabs_user and self.oxylabs_pass)
        return self.fetch_backend == "direct"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def model_id(self, requested: str | None) -> str:
        return requested or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the configuration. Missing credentials fail per request, not at import.

    That keeps ``/health`` callable so an orchestrator can tell "running but not
    configured" apart from "not running".
    """
    return Settings(
        fetch_backend=(os.getenv("FETCH_BACKEND") or "direct").strip().lower(),
        oxylabs_user=os.getenv("OXYLABS_USER") or None,
        oxylabs_pass=os.getenv("OXYLABS_PASS") or None,
        oxylabs_geo_location=os.getenv("OXYLABS_GEO_LOCATION") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        internal_token=os.getenv("LLM_SCRAPER_INTERNAL_TOKEN") or None,
        user_agent=os.getenv("SCRAPER_USER_AGENT") or DEFAULT_USER_AGENT,
        ocr_languages=os.getenv("OCR_LANGUAGES") or "eng+spa",
    )


def get_llm_client() -> AsyncAnthropic:
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY or ANTHROPIC_API_KEY is not configured")
    kwargs: dict[str, str] = {"api_key": settings.llm_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return AsyncAnthropic(**kwargs)
