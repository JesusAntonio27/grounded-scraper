"""Fetch backends.

``direct`` (default) is a plain HTTP GET: free, and enough for static pages and PDFs.
``oxylabs`` sends the URL to the Oxylabs Web Scraper API, which renders JavaScript,
runs browser instructions and can pin a geo-location.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
OXYLABS_URL = "https://realtime.oxylabs.io/v1/queries"


class FetchError(RuntimeError):
    """Expected, source-local fetch failure with a short message that is safe to log."""


def bounded_int(value: object, *, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return min(max(parsed, lower), upper)


async def fetch_html(url: str, options: dict[str, Any]) -> tuple[str, int]:
    """Return ``(html, elapsed_ms)`` using the configured backend."""
    settings = get_settings()
    timeout = bounded_int(options.get("timeout"), default=120, lower=5, upper=300)
    if settings.fetch_backend == "oxylabs":
        return await _fetch_with_oxylabs(url, options, timeout)
    if settings.fetch_backend != "direct":
        raise FetchError(f"Unknown FETCH_BACKEND {settings.fetch_backend!r}")
    if options.get("browser_instructions"):
        logger.warning("browser_instructions need FETCH_BACKEND=oxylabs; ignored for %s", url)
    started = time.monotonic()
    response = await _get(url, timeout)
    if not response.text.strip():
        raise FetchError("Upstream returned empty content")
    return response.text, int((time.monotonic() - started) * 1000)


async def download(url: str, *, timeout: int = 150) -> tuple[bytes, int]:
    """Return ``(body, elapsed_ms)`` with a direct GET (used for PDFs)."""
    started = time.monotonic()
    response = await _get(url, timeout)
    return response.content, int((time.monotonic() - started) * 1000)


async def _get(url: str, timeout: int) -> httpx.Response:
    headers = {"User-Agent": get_settings().user_agent}
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise FetchError(f"Fetch timed out after {timeout}s") from exc
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"Upstream HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Fetch failed: {type(exc).__name__}") from exc
    return response


async def _fetch_with_oxylabs(url: str, options: dict[str, Any], timeout: int) -> tuple[str, int]:
    settings = get_settings()
    if not (settings.oxylabs_user and settings.oxylabs_pass):
        raise FetchError("Oxylabs credentials are not configured")
    payload: dict[str, Any] = {
        "source": "universal",
        "url": url,
        "user_agent_type": options.get("user_agent_type") or "desktop",
    }
    geo_location = options.get("geo_location") or settings.oxylabs_geo_location
    if geo_location:
        payload["geo_location"] = geo_location
    instructions = options.get("browser_instructions")
    if options.get("render_js", True) or instructions:
        payload["render"] = "html"
    if isinstance(instructions, list) and instructions:
        payload["browser_instructions"] = instructions

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout + 10) as client:
            response = await client.post(
                OXYLABS_URL,
                auth=(settings.oxylabs_user, settings.oxylabs_pass),
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise FetchError(f"Oxylabs timeout after {timeout}s") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Oxylabs connection error: {type(exc).__name__}") from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if response.status_code != 200:
        excerpt = " ".join(response.text[:180].split())
        raise FetchError(f"Oxylabs HTTP {response.status_code}: {excerpt}")
    try:
        content = response.json()["results"][0]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise FetchError("Oxylabs returned an unexpected payload") from exc
    if not isinstance(content, str) or not content.strip():
        raise FetchError("Oxylabs returned empty content")
    return content, elapsed_ms
