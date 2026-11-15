"""Catalog parsing, the per-source runner and row building, with a fake API client."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from runner.cli import failure_category, run_source
from runner.client import payload_for, records_from_response
from runner.rows import build_rows, parse_decimal, parse_range
from runner.snapshot import load_source_result, normalize_period, write_source_result
from runner.sources import Catalog, CatalogError, parse_catalog
from tests.conftest import PRICE, RATE


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.modes: list[str | None] = []

    def extract(self, source, fields, *, mode=None) -> dict[str, Any]:
        self.modes.append(mode)
        return self.responses.pop(0)

    def diagnose(self, *args: Any, **kwargs: Any) -> str | None:
        return None  # forces the deterministic fallback


def ok(*records: dict[str, Any]) -> dict[str, Any]:
    return {"status": "extracted", "records": list(records), "meta": {}}


STARTER = {
    "name": "Starter",
    "monthly_price": "19.90",
    "monthly_price_evidence": "$19.90",
    "annual_rate": None,
}
PRO = {
    "name": "Pro plan",
    "monthly_price": 49,
    "monthly_price_evidence": "$49",
    "annual_rate": "30% to 45%",
    "annual_rate_evidence": "30% to 45%",
}


# --- catalog ---------------------------------------------------------------


def test_example_catalog_parses() -> None:
    from runner.sources import load_catalog

    catalog = load_catalog(Path(__file__).parents[1] / "examples" / "sources.example.json")
    assert catalog.source_ids == ("example_pricing_page", "example_fee_brochure")
    assert catalog.get_source("example_fee_brochure").is_pdf


@pytest.mark.parametrize(
    "raw",
    [
        {"fields": [], "sources": []},
        {"fields": [{"name": "price"}, {"name": "price"}]},
        {"fields": [{"name": "price_evidence"}]},
        {
            "fields": [{"name": "price"}],
            "sources": [{"source_id": "a", "url": "ftp://x", "records": ["A"]}],
        },
        {
            "fields": [{"name": "price"}],
            "sources": [
                {"source_id": "a", "url": "https://x", "records": ["A"], "modes": ["vision"]}
            ],
        },
        {
            "fields": [{"name": "price"}],
            "sources": [
                {"source_id": "a", "url": "https://x", "records": ["A"], "aliases": {"B": "C"}}
            ],
        },
        {
            "fields": [{"name": "price"}],
            "sources": [
                {"source_id": "a", "url": "https://x", "records": ["A"]},
                {"source_id": "a", "url": "https://y", "records": ["B"]},
            ],
        },
    ],
)
def test_malformed_catalogs_are_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(CatalogError):
        parse_catalog(raw)


# --- client ----------------------------------------------------------------


def test_records_are_mapped_through_aliases_and_deduplicated(catalog: Catalog) -> None:
    source = catalog.get_source("acme")
    response = ok(STARTER, PRO, {**PRO, "name": "pro"}, {"name": "Enterprise", "monthly_price": 1})
    records = records_from_response(response, source, catalog.fields)
    assert [r["name"] for r in records] == ["Starter", "Pro"]


def test_pdf_sources_use_the_pdf_payload(catalog: Catalog) -> None:
    payload = payload_for(catalog.get_source("brochure"), catalog.fields)
    assert payload["pdf_url"].endswith(".pdf") and "fetch_options" not in payload


# --- runner ----------------------------------------------------------------


def test_incomplete_text_reading_falls_back_to_multimodal(tmp_path: Path, catalog: Catalog) -> None:
    client = FakeClient([ok(STARTER), ok(STARTER, PRO)])
    result = run_source(catalog, "acme", "2026-11-01", "run1", tmp_path, client=client)
    assert client.modes == ["text", "multimodal"]
    assert result["status"] == "success" and result["mode"] == "multimodal"
    # Starter has a price but no rate: success, flagged as partial with a diagnostic.
    assert result["partial_fields"] == [{"name": "Starter", "missing_fields": ["annual_rate"]}]
    assert result["diagnostic_origin"] == "fallback"


def test_every_mode_failing_leaves_a_failed_artifact(tmp_path: Path, catalog: Catalog) -> None:
    client = FakeClient([{"status": "failed", "error": "Upstream HTTP 503"}, ok()])
    result = run_source(catalog, "acme", "2026-11-01", "run1", tmp_path, client=client)
    assert result["status"] == "failed"
    assert result["failure_category"] == "page-changed"  # the second mode returned no records
    stored = load_source_result(tmp_path, "2026-11-01", "run1", catalog.get_source("acme"))
    assert stored is not None and stored["status"] == "failed"


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("text: Fetch timed out after 120s", "upstream-unavailable"),
        ("x: missing records (Pro)", "page-changed"),
        ("weird", "uncertain"),
    ],
)
def test_failure_categories(reason: str, category: str) -> None:
    assert failure_category(reason) == category


# --- snapshot & rows -------------------------------------------------------


def test_period_must_be_first_day_of_month() -> None:
    assert normalize_period("2026-11-01") == "2026-11-01"
    with pytest.raises(ValueError):
        normalize_period("2026-11-15")


def test_artifact_from_another_run_is_not_trusted(tmp_path: Path, catalog: Catalog) -> None:
    source = catalog.get_source("acme")
    write_source_result(
        tmp_path,
        "2026-11-01",
        "run1",
        "acme",
        {"source_id": "acme", "period": "2026-11-01", "run_id": "other"},
    )
    stored = load_source_result(tmp_path, "2026-11-01", "run1", source)
    assert stored is not None and stored["error"] == "Source artifact identity mismatch"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("19.90", Decimal("19.90")),
        ("95.2%", Decimal("95.2")),
        ("$1,299", Decimal("1299")),
        ("12,5", Decimal("12.5")),
        ("30% to 45%", None),
        ("base rate + 5", None),
        (None, None),
    ],
)
def test_parse_decimal_only_accepts_single_literal_numbers(
    raw: object, expected: Decimal | None
) -> None:
    assert parse_decimal(raw) == expected


def test_parse_range() -> None:
    assert parse_range("from 30% to 45%") == (Decimal("30"), Decimal("45"))
    assert parse_range("desde 30 hasta 45") == (Decimal("30"), Decimal("45"))
    assert parse_range("45 - 30") is None


def test_rows_cover_every_source_and_record(tmp_path: Path, catalog: Catalog) -> None:
    run_source(
        catalog, "acme", "2026-11-01", "run1", tmp_path, client=FakeClient([ok(STARTER, PRO)])
    )
    rows = {
        (r["source_id"], r["record_name"]): r
        for r in build_rows(catalog, tmp_path, "2026-11-01", "run1")
    }
    assert set(rows) == {("acme", "Starter"), ("acme", "Pro"), ("brochure", "Card")}

    pro = rows[("acme", "Pro")]
    assert pro["status"] == "captured"
    assert pro["field_values"] == {"monthly_price": Decimal("49"), "annual_rate": None}
    assert pro["field_ranges"] == {"annual_rate": [Decimal("30"), Decimal("45")]}
    assert pro["raw_field_values"]["annual_rate"] == "30% to 45%"

    starter = rows[("acme", "Starter")]
    assert starter["status"] == "captured" and "annual_rate" in starter["diagnostic"]

    # The brochure never ran: its row exists, empty and explicitly unavailable.
    card = rows[("brochure", "Card")]
    assert card["status"] == "unavailable" and card["field_values"] == {
        "monthly_price": None,
        "annual_rate": None,
    }


def test_catalog_fields_shared_with_api() -> None:
    assert (PRICE.evidence_key, RATE.evidence_key) == (
        "monthly_price_evidence",
        "annual_rate_evidence",
    )
