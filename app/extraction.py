"""Grounded extraction: fetch -> readable text (+ OCR) -> LLM JSON -> deterministic evidence check."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.config import get_llm_client, get_settings
from app.evidence import FieldSpec, validate_records
from app.fetch import FetchError, download, fetch_html
from app.images import load_relevant_images, ocr_image
from app.text import (
    EmptyDocumentError,
    diagnostic_excerpt,
    focus_context,
    html_to_text,
    keyword_positions,
)

logger = logging.getLogger(__name__)
MAX_LLM_TOKENS = 3_000


class ExtractionError(RuntimeError):
    """Expected source-local failure, optionally carrying current-page evidence."""

    def __init__(
        self,
        message: str,
        *,
        current_excerpt: str = "",
        partial_output: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.current_excerpt = current_excerpt
        self.partial_output = partial_output

    def add_context(
        self, *, current_excerpt: str, partial_output: dict[str, Any] | None = None
    ) -> ExtractionError:
        if not self.current_excerpt:
            self.current_excerpt = current_excerpt
        if self.partial_output is None and partial_output is not None:
            self.partial_output = partial_output
        return self


@dataclass(frozen=True)
class ExtractionTask:
    """What to extract from one URL: the expected records and their fields."""

    url: str
    records: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    instructions: str = ""
    focus_terms: tuple[str, ...] = ()

    @property
    def marker_terms(self) -> tuple[str, ...]:
        return tuple(keyword for field in self.fields for keyword in field.keywords)

    @property
    def image_keywords(self) -> tuple[str, ...]:
        words = {keyword.lower() for keyword in self.marker_terms if keyword.strip()}
        for record in self.records:
            words.update(word.lower() for word in re.findall(r"\w{4,}", record))
        return tuple(sorted(words))


async def extract_html(
    task: ExtractionTask, fetch_options: dict[str, Any], llm_options: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        html, fetch_ms = await fetch_html(task.url, fetch_options)
    except FetchError as exc:
        raise ExtractionError(str(exc)) from exc
    mode = str(llm_options.get("mode") or "text")
    try:
        text = html_to_text(html)
    except EmptyDocumentError as exc:
        if mode != "multimodal":
            raise ExtractionError(str(exc)) from exc
        text = ""
    images: list[dict[str, Any]] = []
    image_text = ""
    if mode == "multimodal":
        images, image_text = await load_relevant_images(task.url, html, task.image_keywords)
    source_text = f"{text}\n\nOCR OF CURRENT PAGE IMAGES:\n{image_text}" if image_text else text
    evidence_text = focus_context(source_text, task.focus_terms, task.marker_terms)
    if not evidence_text.strip():
        raise ExtractionError("Page has no readable text or OCR output")
    try:
        data, llm_ms = await _extract_with_llm(
            evidence_text, task, llm_options.get("model"), images
        )
    except ExtractionError as exc:
        exc.add_context(current_excerpt=diagnostic_excerpt(evidence_text, task.marker_terms))
        raise
    return {
        "status": "extracted",
        "records": data["records"],
        "meta": {
            "fetch_ms": fetch_ms,
            "llm_ms": llm_ms,
            "total_ms": int((time.monotonic() - started) * 1000),
            "mode": mode,
            "images_sent": len(images),
            "images_with_ocr": image_text.count("[Image "),
            "diagnostic_excerpt": diagnostic_excerpt(evidence_text, task.marker_terms),
        },
    }


async def extract_pdf(task: ExtractionTask, llm_options: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        content, fetch_ms = await download(task.url)
    except FetchError as exc:
        raise ExtractionError(str(exc)) from exc
    text = _pdf_text(content)
    if _needs_ocr(text, task.fields):
        ocr_text = await asyncio.to_thread(_ocr_pdf_pages, content)
        if ocr_text:
            text = f"{text}\n\nOCR OF CURRENT PDF PAGES:\n{ocr_text}"
    if not text.strip():
        raise ExtractionError("PDF has no extractable text")
    context = focus_context(text, task.focus_terms, task.marker_terms)
    try:
        data, llm_ms = await _extract_with_llm(context, task, llm_options.get("model"), [])
    except ExtractionError as exc:
        exc.add_context(current_excerpt=diagnostic_excerpt(context, task.marker_terms))
        raise
    return {
        "status": "extracted",
        "records": data["records"],
        "meta": {
            "fetch_ms": fetch_ms,
            "llm_ms": llm_ms,
            "total_ms": int((time.monotonic() - started) * 1000),
            "mode": "pdf_text",
            "images_sent": 0,
            "diagnostic_excerpt": diagnostic_excerpt(context, task.marker_terms),
        },
    }


def _pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - any parser failure is a source failure
        raise ExtractionError("PDF could not be parsed") from exc


def _needs_ocr(text: str, fields: tuple[FieldSpec, ...]) -> bool:
    """OCR when the text layer is empty or misses a field's keywords.

    Rate tables in PDFs are often vector drawings or images, so the text layer
    can contain the whole brochure except the one table that matters.
    """
    if not text.strip():
        return True
    return any(field.keywords and not keyword_positions(text, field.keywords) for field in fields)


def _ocr_pdf_pages(content: bytes) -> str:
    """Render the PDF with poppler's ``pdftoppm`` and OCR each page."""
    try:
        with tempfile.TemporaryDirectory(prefix="grounded-pdf-") as directory:
            directory_path = Path(directory)
            pdf_path = directory_path / "source.pdf"
            pdf_path.write_bytes(content)
            rendered = subprocess.run(
                ["pdftoppm", "-r", "300", "-png", str(pdf_path), str(directory_path / "page")],
                check=False,
                capture_output=True,
                timeout=90,
            )
            if rendered.returncode:
                logger.warning("PDF rendering for OCR failed: %s", rendered.stderr[:160])
                return ""
            chunks = [
                ocr_image(image_path.read_bytes())
                for image_path in sorted(directory_path.glob("page-*.png"))
            ]
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("PDF OCR unavailable: %s", type(exc).__name__)
        return ""
    return "\n\n".join(chunk for chunk in chunks if chunk)


def build_schema(task: ExtractionTask) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "One of the requested record names, written as in the request",
        }
    }
    for field in task.fields:
        value_types = ["number", "string", "null"] if field.kind == "number" else ["string", "null"]
        properties[field.name] = {"type": value_types}
        properties[field.evidence_key] = {
            "type": ["string", "null"],
            "description": f"Short literal quote from the current content that contains the {field.name} value",
        }
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {"type": "object", "properties": properties, "required": list(properties)},
            }
        },
        "required": ["records"],
    }


def build_prompt(task: ExtractionTask, text: str) -> str:
    record_list = "\n".join(f"- {record}" for record in task.records)
    instructions = (
        f"Source-specific instructions:\n{task.instructions}\n\n"
        if task.instructions.strip()
        else ""
    )
    return (
        "You are a data extractor. Reply only with valid JSON that matches the schema. "
        "Do not invent values and do not use prior knowledge or historical values. "
        f"Today is {date.today().isoformat()}; if the source shows several validity periods, "
        "use only the one in force today, and use null if you cannot tell which one that is. "
        "For every non-null field, put in its *_evidence field a short literal quote from the "
        "current content that contains the same value. If you cannot quote it exactly, set both "
        "the value and its evidence to null.\n\n"
        f"Records to extract:\n{record_list}\n\n"
        f"{instructions}"
        f"Schema:\n{json.dumps(build_schema(task), ensure_ascii=False)}\n\n"
        f"Content:\n{text}"
    )


async def _extract_with_llm(
    text: str, task: ExtractionTask, model: str | None, images: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    settings = get_settings()
    if not settings.llm_configured:
        raise ExtractionError("LLM credentials are not configured")
    prompt = build_prompt(task, text)
    content: str | list[dict[str, Any]] = (
        [*images, {"type": "text", "text": prompt}] if images else prompt
    )
    started = time.monotonic()
    try:
        client = get_llm_client()
        response = await client.messages.create(
            model=settings.model_id(model),
            max_tokens=MAX_LLM_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:  # noqa: BLE001 - the SDK raises many types; report the class only
        raise ExtractionError(f"LLM extraction failed: {type(exc).__name__}") from exc
    parsed = parse_json(response_text(response))
    _validate(parsed, text, task)

    if images and has_missing_fields(parsed, task.fields):
        # A second, independent reading of the images often recovers values the
        # first pass skipped. Its output goes through the same validation.
        retry_prompt = (
            f"{prompt}\n\nThe first reading was partial:\n{json.dumps(parsed, ensure_ascii=False)}\n\n"
            "Do a second, independent reading of the current images and OCR text. Fill missing "
            "fields only if you can return a literal quote with the same value. Keep the valid "
            "values from the first reading and reply again with JSON only."
        )
        try:
            retry_response = await client.messages.create(
                model=settings.model_id(model),
                max_tokens=MAX_LLM_TOKENS,
                messages=[
                    {"role": "user", "content": [*images, {"type": "text", "text": retry_prompt}]}
                ],
            )
            retried = parse_json(response_text(retry_response))
            _validate(retried, text, task)
            merge_missing_records(parsed, retried, task.fields)
        except Exception as exc:  # noqa: BLE001 - the first reading already stands on its own
            logger.warning("second image reading failed: %s", type(exc).__name__)
    return parsed, int((time.monotonic() - started) * 1000)


def _validate(data: dict[str, Any], text: str, task: ExtractionTask) -> None:
    try:
        validate_records(data, text, task.fields)
    except ValueError as exc:
        raise ExtractionError(str(exc), partial_output=data) from exc


def has_missing_fields(data: dict[str, Any], fields: tuple[FieldSpec, ...]) -> bool:
    records = data.get("records")
    return isinstance(records, list) and any(
        isinstance(record, dict) and any(record.get(field.name) is None for field in fields)
        for record in records
    )


def merge_missing_records(
    original: dict[str, Any], retried: dict[str, Any], fields: tuple[FieldSpec, ...]
) -> None:
    """Fill only the gaps of ``original`` with validated values from ``retried``."""
    original_records = original.get("records")
    retried_records = retried.get("records")
    if not isinstance(original_records, list) or not isinstance(retried_records, list):
        return
    by_name = {
        str(record.get("name")).casefold(): record
        for record in original_records
        if isinstance(record, dict) and record.get("name")
    }
    for candidate in retried_records:
        if not isinstance(candidate, dict) or not candidate.get("name"):
            continue
        current = by_name.get(str(candidate["name"]).casefold())
        if current is None:
            original_records.append(candidate)
            continue
        for field in fields:
            if current.get(field.name) is None and candidate.get(field.name) is not None:
                current[field.name] = candidate[field.name]
                current[field.evidence_key] = candidate.get(field.evidence_key)


def response_text(response: Any) -> str:
    blocks = getattr(response, "content", [])
    return "".join(getattr(block, "text", "") for block in blocks if getattr(block, "text", None))


def parse_json(raw: str) -> dict[str, Any]:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ExtractionError("LLM returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExtractionError("LLM returned a JSON value instead of an object")
    return value


async def diagnose_failure(
    source_url: str,
    failure_reason: str,
    current_excerpt: str = "",
    partial_output: dict[str, Any] | None = None,
) -> str:
    """One short, human-readable log line explaining a failure.

    Falls back to a deterministic message whenever the LLM is unavailable, so a
    diagnostic never hides the root failure.
    """
    fallback = technical_diagnostic(failure_reason)
    settings = get_settings()
    if not settings.llm_configured:
        return fallback
    prompt = (
        "Write one short sentence (at most 160 characters) for an operations log that "
        "explains why a source could not be extracted. Do not invent causes, do not mention "
        "secrets and do not suggest fixes.\n"
        f"Source: {source_url}\nTechnical failure: {failure_reason[:400]}\n"
        f"Current excerpt: {current_excerpt[:800]}\n"
        f"Partial output: {json.dumps(partial_output or {}, ensure_ascii=False)[:300]}"
    )
    try:
        client = get_llm_client()
        response = await client.messages.create(
            model=settings.model_id(None),
            max_tokens=90,
            messages=[{"role": "user", "content": prompt}],
        )
        clean = " ".join(response_text(response).split())[:180]
        return clean or fallback
    except Exception as exc:  # noqa: BLE001 - deliberate fallback
        logger.info("diagnostic LLM unavailable: %s", type(exc).__name__)
        return fallback


def technical_diagnostic(reason: str) -> str:
    concise = " ".join(reason.split())[:160]
    return f"The source could not be processed: {concise or 'technical failure without detail.'}"
