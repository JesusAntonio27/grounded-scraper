"""Pick the page images most likely to hold the requested values, and OCR them.

OCR is deliberately independent of the LLM. A value that is visible only inside an
image survives validation only if its digits also appear in this OCR text, so the
model cannot "read" a number that nothing else can confirm.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import mimetypes
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5_000_000
SUPPORTED_IMAGE_MIME_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}


async def load_relevant_images(
    page_url: str, html: str, keywords: tuple[str, ...], *, max_images: int = MAX_IMAGES
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(image_blocks_for_the_llm, ocr_text)`` for the best candidate images."""
    images: list[dict[str, Any]] = []
    ocr_chunks: list[str] = []
    headers = {"User-Agent": get_settings().user_agent}
    async with httpx.AsyncClient(timeout=40, follow_redirects=True, headers=headers) as client:
        supplemental_html = await zendesk_article_body(client, page_url)
        for image_url in image_urls(page_url, f"{html}\n{supplemental_html}", keywords)[
            :max_images
        ]:
            try:
                response = await client.get(image_url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            mime_type = response.headers.get("content-type", "").split(";", 1)[0]
            if not mime_type.startswith("image/"):
                mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                logger.warning("unsupported image type url=%s type=%s", image_url, mime_type)
                continue
            if len(response.content) > MAX_IMAGE_BYTES:
                continue
            ocr_text = await asyncio.to_thread(ocr_image, response.content)
            if ocr_text:
                ocr_chunks.append(f"[Image {image_url}]\n{ocr_text}")
            images.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64.b64encode(response.content).decode("ascii"),
                    },
                }
            )
    return images, "\n\n".join(ocr_chunks)


def ocr_image(content: bytes) -> str:
    """OCR one image; any failure returns ``""`` so validation stays conservative."""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            # Value banners are usually tables; one uniform block (--psm 6) is
            # far more reliable than Tesseract's automatic layout for them.
            text = pytesseract.image_to_string(
                image, lang=get_settings().ocr_languages, config="--psm 6"
            )
    except Exception as exc:  # noqa: BLE001 - OCR is best-effort by design
        logger.warning("image OCR failed: %s", type(exc).__name__)
        return ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:20_000]


def image_urls(page_url: str, html: str, keywords: tuple[str, ...]) -> list[str]:
    """Candidate image URLs, best first.

    Help-center article attachments come first (they are the content). Theme
    assets and logos are skipped. Other images must mention a keyword in their
    URL, alt text, class or surrounding label.
    """
    soup = BeautifulSoup(html, "html.parser")
    lowered_keywords = tuple(keyword.lower() for keyword in keywords if keyword.strip())
    candidates: list[tuple[str, str]] = []
    for image in soup.find_all("img"):
        label = " ".join(
            str(image.get(attribute) or "")
            for attribute in ("alt", "title", "aria-label", "id", "class")
        )
        candidates.extend(
            (str(value), label)
            for value in (
                image.get("src"),
                image.get("data-src"),
                image.get("data-lazy-src"),
                image.get("data-original"),
            )
            if value
        )
        for attribute in ("srcset", "data-srcset"):
            candidates.extend(
                (part.strip().split(" ")[0], label)
                for part in str(image.get(attribute) or "").split(",")
                if part.strip()
            )
    for source in soup.find_all("source"):
        picture = source.find_parent("picture")
        fallback = picture.find("img") if picture else None
        label = "picture source " + (str(fallback.get("alt") or "") if fallback else "")
        for attribute in ("srcset", "data-srcset"):
            candidates.extend(
                (part.strip().split(" ")[0], label)
                for part in str(source.get(attribute) or "").split(",")
                if part.strip()
            )
    for anchor in soup.select("a[href*='article_attachments']"):
        candidates.append((str(anchor.get("href")), "article attachment"))
    for tag in soup.find_all(style=True):
        label = " ".join(
            (
                "background image",
                str(tag.get("id") or ""),
                str(tag.get("class") or ""),
                tag.get_text(" ", strip=True)[:160],
            )
        )
        for raw in re.findall(r"url\((['\"]?)(.*?)\1\)", str(tag.get("style")), flags=re.I):
            candidates.append((raw[1], label))

    preferred: list[str] = []
    values: list[str] = []
    for src, label in candidates:
        if not src or src.startswith("data:"):
            continue
        combined = f"{src} {label}".lower()
        resolved = urljoin(page_url, src)
        if "/article_attachments/" in resolved:
            if resolved not in preferred:
                preferred.append(resolved)
            continue
        if "/theming_assets/" in resolved or "logo" in combined:
            continue
        if lowered_keywords and not any(keyword in combined for keyword in lowered_keywords):
            continue
        if resolved not in values:
            values.append(resolved)
    return [*preferred, *values]


async def zendesk_article_body(client: httpx.AsyncClient, page_url: str) -> str:
    """Fetch a Zendesk help-center article body through its public JSON API.

    Help centers often lazy-load attachments, so the rendered page can miss the
    image that holds the table.
    """
    api_url = zendesk_article_api_url(page_url)
    if api_url is None:
        return ""
    try:
        response = await client.get(api_url)
        response.raise_for_status()
        article = response.json().get("article")
    except (httpx.HTTPError, ValueError, AttributeError):
        return ""
    if not isinstance(article, dict):
        return ""
    body = article.get("body")
    return body if isinstance(body, str) else ""


def zendesk_article_api_url(page_url: str) -> str | None:
    parsed = urlparse(page_url)
    match = re.search(r"/hc/([^/]+)/articles/(\d+)", parsed.path)
    if not parsed.scheme or not parsed.netloc or match is None:
        return None
    locale, article_id = match.groups()
    return (
        f"{parsed.scheme}://{parsed.netloc}/api/v2/help_center/{locale}/articles/{article_id}.json"
    )
