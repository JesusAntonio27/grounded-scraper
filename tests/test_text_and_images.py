from __future__ import annotations

import pytest

from app.images import image_urls, zendesk_article_api_url
from app.text import (
    MAX_TEXT_CHARS,
    EmptyDocumentError,
    focus_context,
    html_to_text,
    keyword_positions,
)


def test_html_to_text_keeps_styled_values_and_drops_scripts() -> None:
    html = """<html><body><nav>Home</nav><script>var price = 999;</script>
    <div class="card"><h3>Pro</h3><div class="amount">$49</div></div></body></html>"""
    text = html_to_text(html)
    assert "$49" in text and "Pro" in text
    assert "999" not in text


def test_html_without_text_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        html_to_text("<html><body><script>x()</script></body></html>")


def test_keyword_positions_tolerates_letter_spacing() -> None:
    assert keyword_positions("intro ... A P R 30%", ("APR",)) == [10]


def test_short_text_is_returned_unchanged() -> None:
    assert focus_context("short page", ("Pro",), ("price",)) == "short page"


def test_long_text_keeps_windows_around_focus_terms_within_budget() -> None:
    filler = "navigation link " * 10_000
    text = f"{filler} Pro plan costs $49 per month {filler}"
    context = focus_context(text, ("Pro plan",), ("price",))
    assert len(context) <= MAX_TEXT_CHARS
    assert "Pro plan costs $49" in context


def test_image_urls_prefers_attachments_skips_logos_and_needs_keywords() -> None:
    html = """
    <img src="/assets/logo-price.png" alt="logo">
    <img src="/img/team.jpg" alt="our team">
    <img src="/img/price-table.png" alt="Price table">
    <a href="https://help.example.com/hc/article_attachments/123">table</a>
    """
    urls = image_urls("https://example.com/pricing", html, ("price",))
    assert urls == [
        "https://help.example.com/hc/article_attachments/123",
        "https://example.com/img/price-table.png",
    ]


def test_zendesk_api_url_is_derived_from_an_article_url() -> None:
    assert (
        zendesk_article_api_url("https://help.example.com/hc/es/articles/42-fees")
        == "https://help.example.com/api/v2/help_center/es/articles/42.json"
    )
    assert zendesk_article_api_url("https://example.com/pricing") is None
