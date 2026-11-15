from __future__ import annotations

import pytest

from app.evidence import FieldSpec, is_supported, normalize_text, numeric_token, validate_records
from tests.conftest import PRICE, RATE, SUMMARY

SOURCE = "Pricing\nStarter plan: $19.90 per month\nPro plan: $49 per month\nAPR rate 36.5%"


@pytest.mark.parametrize(
    ("token", "expected"),
    [("19.90", "199"), ("19,9", "199"), ("1,234.50", "12345"), ("49", "49"), ("36.50", "365")],
)
def test_numeric_token_ignores_superficial_formatting(token: str, expected: str) -> None:
    assert numeric_token(token) == expected


def test_normalize_text_strips_accents_case_and_spacing() -> None:
    assert normalize_text("  Crédito   PERSONAL ") == "credito personal"


def test_number_backed_by_quote_and_source_is_supported() -> None:
    assert is_supported(19.9, "Starter plan: $19.90 per month", SOURCE, PRICE)


def test_thousands_separator_does_not_reject_a_correct_value() -> None:
    page = "Enterprise plan: $1,000 per month"
    assert is_supported(1000, "$1,000 per month", page, PRICE)
    assert is_supported("1000", "$1.000 per month", "Enterprise: $1.000 per month", PRICE)
    assert not is_supported(1001, "$1,000 per month", page, PRICE)


def test_number_absent_from_the_source_is_rejected() -> None:
    # The model "remembers" an old price; the quote even contains it, the page does not.
    assert not is_supported(24.9, "Starter plan: $24.90 per month", SOURCE, PRICE)


def test_number_missing_from_its_own_quote_is_rejected() -> None:
    assert not is_supported(49, "Starter plan: $19.90 per month", SOURCE, PRICE)


def test_missing_or_empty_quote_is_rejected() -> None:
    assert not is_supported(49, None, SOURCE, PRICE)
    assert not is_supported(49, "   ", SOURCE, PRICE)


def test_field_keyword_must_appear_in_the_source() -> None:
    no_rate_page = "Starter plan: $19.90 per month (36.5 stars)"
    assert not is_supported("36.5", "36.5 stars", no_rate_page, RATE)


def test_letter_spaced_pdf_label_still_counts_as_a_mention() -> None:
    spaced = "R A T E   36.5 %"
    assert is_supported("36.5", "36.5 %", spaced, FieldSpec("annual_rate", "number", ("rate",)))


def test_text_value_must_be_in_quote_and_quote_in_source() -> None:
    assert is_supported(
        "unlimited users",
        "Pro plan includes unlimited users",
        "The Pro plan includes unlimited users.",
        SUMMARY,
    )
    assert not is_supported(
        "unlimited users", "Pro plan includes unlimited users", "Pro plan: 5 users.", SUMMARY
    )


def test_validate_records_nulls_only_unsupported_values() -> None:
    data = {
        "records": [
            {
                "name": "Starter",
                "monthly_price": 19.9,
                "monthly_price_evidence": "Starter plan: $19.90 per month",
                "annual_rate": 99,
                "annual_rate_evidence": "rate 99%",
            }
        ]
    }
    rejected = validate_records(data, SOURCE, (PRICE, RATE))
    record = data["records"][0]
    assert rejected == 1
    assert record["monthly_price"] == 19.9
    assert record["annual_rate"] is None and record["annual_rate_evidence"] is None


def test_validate_records_rejects_a_malformed_response() -> None:
    with pytest.raises(ValueError):
        validate_records({"items": []}, SOURCE, (PRICE,))
    with pytest.raises(ValueError):
        validate_records({"records": [{"monthly_price": 1}]}, SOURCE, (PRICE,))
