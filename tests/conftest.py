"""Shared fixtures. Every test is offline: no network, no LLM, no database."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.evidence import FieldSpec
from runner.sources import Catalog, SourceSpec

PRICE = FieldSpec("monthly_price", "number", ("price", "$"))
RATE = FieldSpec("annual_rate", "number", ("rate", "APR"))
SUMMARY = FieldSpec("plan_summary", "text", ())


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch: pytest.MonkeyPatch):
    for name in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "LLM_SCRAPER_INTERNAL_TOKEN", "FETCH_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        fields=(PRICE, RATE),
        sources=(
            SourceSpec(
                source_id="acme",
                url="https://example.com/pricing",
                records=("Starter", "Pro"),
                modes=("text", "multimodal"),
                aliases=(("Pro plan", "Pro"),),
            ),
            SourceSpec(source_id="brochure", url="https://example.com/fees.pdf", records=("Card",)),
        ),
    )
