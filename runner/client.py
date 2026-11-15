"""Standard-library HTTP client for the extraction API (no extra dependencies in the worker)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.evidence import FieldSpec, normalize_text
from runner.sources import SourceSpec


class ApiRequestError(RuntimeError):
    """A failed API request, with a deliberately short message that is safe to log."""


@dataclass(frozen=True)
class ScraperApiClient:
    base_url: str
    timeout_seconds: int = 180
    internal_token: str | None = None

    @classmethod
    def from_environment(cls) -> ScraperApiClient:
        return cls(
            base_url=os.getenv("LLM_SCRAPER_BASE_URL", "http://127.0.0.1:8000"),
            timeout_seconds=int(os.getenv("LLM_SCRAPER_TIMEOUT_SECONDS", "180")),
            internal_token=os.getenv("LLM_SCRAPER_INTERNAL_TOKEN") or None,
        )

    def extract(
        self, source: SourceSpec, fields: tuple[FieldSpec, ...], *, mode: str | None = None
    ) -> dict[str, Any]:
        endpoint = "/extract/pdf" if source.is_pdf else "/extract"
        return self._post_json(endpoint, payload_for(source, fields, mode=mode))

    def diagnose(
        self,
        source_url: str,
        failure_reason: str,
        *,
        current_excerpt: str = "",
        partial_output: dict[str, Any] | None = None,
    ) -> str | None:
        try:
            response = self._post_json(
                "/diagnose",
                {
                    "source_url": source_url,
                    "failure_reason": failure_reason[:500],
                    "current_excerpt": current_excerpt[:1_200],
                    "partial_output": partial_output,
                },
            )
        except ApiRequestError:
            return None
        diagnostic = response.get("diagnostic")
        return diagnostic if isinstance(diagnostic, str) and diagnostic.strip() else None

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.internal_token:
            headers["X-LLM-Scraper-Token"] = self.internal_token
        request = urllib.request.Request(
            self.base_url.rstrip("/") + endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            excerpt = exc.read(300).decode("utf-8", errors="replace").replace("\n", " ")
            raise ApiRequestError(f"API HTTP {exc.code}: {excerpt[:220]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiRequestError(f"API connection failed: {type(exc).__name__}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiRequestError("API returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ApiRequestError("API returned an unexpected payload")
        return decoded


def payload_for(
    source: SourceSpec, fields: tuple[FieldSpec, ...], *, mode: str | None = None
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "records": list(source.records),
        "fields": [
            {"name": spec.name, "kind": spec.kind, "keywords": list(spec.keywords)}
            for spec in fields
        ],
        "instructions": source.instructions,
        "focus_terms": list(source.focus_terms),
        "llm_options": {"mode": mode or source.modes[0]},
    }
    if source.is_pdf:
        return {"pdf_url": source.url, **common}
    return {
        "url": source.url,
        **common,
        "fetch_options": {
            "render_js": source.render_js,
            "timeout": source.timeout_seconds,
            "browser_instructions": [dict(step) for step in source.browser_instructions],
        },
    }


def records_from_response(
    response: dict[str, Any], source: SourceSpec, fields: tuple[FieldSpec, ...]
) -> list[dict[str, Any]]:
    """Keep only the requested records, mapped to their canonical names, once each."""
    records = response.get("records")
    if not isinstance(records, list):
        return []
    requested = {normalize_text(name): name for name in source.records}
    requested.update({normalize_text(alias): canonical for alias, canonical in source.aliases})
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            continue
        canonical = requested.get(normalize_text(record["name"]))
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        item: dict[str, Any] = {"name": canonical}
        for spec in fields:
            item[spec.name] = record.get(spec.name)
            item[spec.evidence_key] = record.get(spec.evidence_key)
        normalized.append(item)
    return normalized
