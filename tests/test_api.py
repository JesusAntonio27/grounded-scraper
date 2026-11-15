"""The HTTP contract, with the fetcher and the LLM replaced by fakes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.extraction as extraction
from app.main import app

PAGE = "<html><body><h3>Starter</h3><p>$19.90 per month</p><h3>Pro plan</h3><p>$49 per month</p></body></html>"
REQUEST = {
    "url": "https://example.com/pricing",
    "records": ["Starter", "Pro"],
    "fields": [{"name": "monthly_price", "kind": "number", "keywords": ["month", "$"]}],
}


class FakeMessages:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(self.reply))])


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    messages = FakeMessages(
        {
            "records": [
                {
                    "name": "Starter",
                    "monthly_price": 19.9,
                    "monthly_price_evidence": "$19.90 per month",
                },
                # Hallucinated: 59 is not on the page.
                {"name": "Pro", "monthly_price": 59, "monthly_price_evidence": "$59 per month"},
            ]
        }
    )
    monkeypatch.setattr(extraction, "get_llm_client", lambda: SimpleNamespace(messages=messages))

    async def fake_fetch(url: str, options: dict[str, Any]) -> tuple[str, int]:
        return PAGE, 5

    monkeypatch.setattr(extraction, "fetch_html", fake_fetch)
    return messages


def test_health_reports_degraded_without_credentials() -> None:
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded" and body["llm"] == "missing_credentials"


def test_extract_keeps_quoted_values_and_nulls_hallucinated_ones(fake_llm: FakeMessages) -> None:
    body = TestClient(app).post("/extract", json=REQUEST).json()
    assert body["status"] == "extracted"
    by_name = {record["name"]: record for record in body["records"]}
    assert by_name["Starter"]["monthly_price"] == 19.9
    assert by_name["Pro"]["monthly_price"] is None
    prompt = fake_llm.calls[0]["messages"][0]["content"]
    assert "- Starter" in prompt and "$49 per month" in prompt


def test_source_failure_is_a_200_with_status_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_fetch(url: str, options: dict[str, Any]) -> tuple[str, int]:
        raise extraction.FetchError("Upstream HTTP 503")

    monkeypatch.setattr(extraction, "fetch_html", broken_fetch)
    response = TestClient(app).post("/extract", json=REQUEST)
    assert response.status_code == 200
    assert response.json()["status"] == "failed" and "503" in response.json()["error"]


def test_internal_token_is_enforced_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SCRAPER_INTERNAL_TOKEN", "shared-secret-for-tests")
    client = TestClient(app)
    assert client.post("/extract", json=REQUEST).status_code == 401
    assert (
        client.post(
            "/diagnose",
            json={"source_url": "https://example.com", "failure_reason": "x"},
            headers={"X-LLM-Scraper-Token": "shared-secret-for-tests"},
        ).status_code
        == 200
    )


@pytest.mark.parametrize(
    "bad",
    [
        {**REQUEST, "url": "file:///etc/passwd"},
        {**REQUEST, "fields": [{"name": "name"}]},
        {**REQUEST, "fields": [{"name": "price"}, {"name": "price"}]},
        {**REQUEST, "records": []},
    ],
)
def test_invalid_requests_are_rejected(bad: dict[str, Any]) -> None:
    assert TestClient(app).post("/extract", json=bad).status_code == 422
