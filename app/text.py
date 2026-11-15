"""Turn fetched HTML into readable text and keep a long page within the model's budget."""

from __future__ import annotations

import re
from html import unescape

import trafilatura
from bs4 import BeautifulSoup

MAX_TEXT_CHARS = 90_000
_WINDOW_BEFORE = 1_100
_WINDOW_AFTER = 2_200
_HEAD_CHARS = 8_000


class EmptyDocumentError(ValueError):
    """The document has no readable text."""


def html_to_text(html: str) -> str:
    """Combine trafilatura's readable Markdown with the page's structural text.

    Values on JavaScript-heavy pages often live in styled ``div``/``h3`` blocks that
    trafilatura drops as boilerplate. Keeping the structural text as well stops a
    short, navigation-only Markdown result from hiding them.
    """
    markdown = (
        trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            include_images=False,
            output_format="markdown",
            favor_recall=True,
        )
        or ""
    )
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "svg", "noscript", "iframe"]):
        tag.decompose()
    structural = soup.get_text("\n", strip=True)
    structural = re.sub(r"\n{3,}", "\n\n", unescape(structural))
    parts = [part.strip() for part in (markdown, structural) if part.strip()]
    if not parts:
        raise EmptyDocumentError("HTML contains no readable text")
    return "\n\n".join(parts)


def _term_pattern(term: str) -> str | None:
    term = term.strip()
    if not term:
        return None
    if " " in term or len(term) > 24:
        return re.escape(term)
    # Tolerate PDF letter-spacing ("C A T") for single words.
    spaced = r"\s*".join(re.escape(char) for char in term)
    if term[0].isalnum() and term[-1].isalnum():
        return rf"(?<!\w){spaced}(?!\w)"
    return spaced


def keyword_positions(text: str, terms: tuple[str, ...]) -> list[int]:
    """Start offsets of every match of ``terms``, in document order."""
    patterns = [pattern for pattern in map(_term_pattern, terms) if pattern]
    if not patterns:
        return []
    combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE)
    return [match.start() for match in combined.finditer(text)]


def focus_context(text: str, focus_terms: tuple[str, ...], marker_terms: tuple[str, ...]) -> str:
    """Return ``text`` unchanged if it fits; otherwise windows around relevant terms.

    ``focus_terms`` (record names, section titles) are searched first, then the
    field keywords. The beginning of the document is appended because headers and
    validity dates usually live there.
    """
    if len(text) <= MAX_TEXT_CHARS:
        return text
    focus_positions = [
        match.start()
        for term in focus_terms
        if len(term.strip()) >= 4
        for match in re.finditer(re.escape(term.strip()), text, flags=re.IGNORECASE)
    ]
    positions = [*focus_positions[:30], *keyword_positions(text, marker_terms)[:40]]
    windows = [text[max(0, found - _WINDOW_BEFORE) : found + _WINDOW_AFTER] for found in positions]
    return ("\n\n".join(windows) + "\n\n" + text[:_HEAD_CHARS])[:MAX_TEXT_CHARS]


def diagnostic_excerpt(text: str, marker_terms: tuple[str, ...]) -> str:
    """Short excerpt around the first relevant term, used only to explain failures."""
    positions = keyword_positions(text, marker_terms)
    start = max(0, positions[0] - 300) if positions else 0
    return " ".join(text[start : start + 1_200].split())
