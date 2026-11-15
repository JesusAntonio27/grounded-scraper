"""The source catalog: which URLs to read, which records each one must yield, and which fields.

The catalog is a JSON file (see ``examples/sources.example.json``). Each source is an
isolated task: one broken page never takes the others down with it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.evidence import FieldSpec

Mode = Literal["text", "multimodal"]
VALID_MODES = ("text", "multimodal")
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")
FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


class CatalogError(ValueError):
    """The catalog file is malformed."""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: str
    records: tuple[str, ...]
    # Extraction attempts inside the same task, in order. Airflow retries the
    # whole task with backoff only when every mode has failed.
    modes: tuple[Mode, ...] = ("text",)
    instructions: str = ""
    render_js: bool = True
    timeout_seconds: int = 120
    browser_instructions: tuple[dict[str, Any], ...] = ()
    # Alternate labels seen on the page, mapped to a canonical record name.
    # Aliases never create extra rows.
    aliases: tuple[tuple[str, str], ...] = ()
    focus_terms: tuple[str, ...] = ()

    @property
    def is_pdf(self) -> bool:
        return self.url.lower().split("?", 1)[0].endswith(".pdf")


@dataclass(frozen=True)
class Catalog:
    fields: tuple[FieldSpec, ...]
    sources: tuple[SourceSpec, ...] = field(default_factory=tuple)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    def get_source(self, source_id: str) -> SourceSpec:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"unknown source_id {source_id!r}")


def load_catalog(path: str | Path) -> Catalog:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read catalog {path}: {type(exc).__name__}") from exc
    return parse_catalog(raw)


def parse_catalog(raw: object) -> Catalog:
    if not isinstance(raw, dict):
        raise CatalogError("catalog must be a JSON object")
    fields = tuple(_parse_field(item) for item in _list(raw, "fields"))
    if not fields:
        raise CatalogError("catalog needs at least one field")
    names = [spec.name for spec in fields]
    if len(names) != len(set(names)):
        raise CatalogError("field names must be unique")
    sources = tuple(_parse_source(item) for item in _list(raw, "sources"))
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise CatalogError("source_id values must be unique")
    return Catalog(fields=fields, sources=sources)


def _list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise CatalogError(f"{key} must be a list")
    return value


def _parse_field(item: object) -> FieldSpec:
    if not isinstance(item, dict):
        raise CatalogError("each field must be an object")
    name = item.get("name")
    if not isinstance(name, str) or not FIELD_NAME_PATTERN.fullmatch(name):
        raise CatalogError(f"invalid field name {name!r}")
    if name == "name" or name.endswith("_evidence"):
        raise CatalogError(f"field name {name!r} is reserved")
    kind = item.get("kind", "number")
    if kind not in ("number", "text"):
        raise CatalogError(f"field {name!r}: kind must be 'number' or 'text'")
    keywords = item.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise CatalogError(f"field {name!r}: keywords must be a list of strings")
    return FieldSpec(name=name, kind=kind, keywords=tuple(k for k in keywords if k.strip()))


def _parse_source(item: object) -> SourceSpec:
    if not isinstance(item, dict):
        raise CatalogError("each source must be an object")
    source_id = item.get("source_id")
    if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise CatalogError(f"invalid source_id {source_id!r}")
    url = item.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise CatalogError(f"{source_id}: url must start with http:// or https://")
    records = item.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(r, str) and r.strip() for r in records)
        or len(set(records)) != len(records)
    ):
        raise CatalogError(f"{source_id}: records must be a non-empty list of unique names")
    modes = item.get("modes", ["text"])
    if not isinstance(modes, list) or not modes or any(mode not in VALID_MODES for mode in modes):
        raise CatalogError(f"{source_id}: modes must be a non-empty subset of {VALID_MODES}")
    aliases = item.get("aliases", {})
    if not isinstance(aliases, dict) or any(target not in records for target in aliases.values()):
        raise CatalogError(f"{source_id}: every alias must point to one of its records")
    timeout = item.get("timeout_seconds", 120)
    if not isinstance(timeout, int) or not 5 <= timeout <= 300:
        raise CatalogError(f"{source_id}: timeout_seconds must be between 5 and 300")
    instructions = item.get("browser_instructions", [])
    if not isinstance(instructions, list) or not all(isinstance(i, dict) for i in instructions):
        raise CatalogError(f"{source_id}: browser_instructions must be a list of objects")
    return SourceSpec(
        source_id=source_id,
        url=url,
        records=tuple(records),
        modes=tuple(modes),
        instructions=str(item.get("instructions", "")),
        render_js=bool(item.get("render_js", True)),
        timeout_seconds=timeout,
        browser_instructions=tuple(instructions),
        aliases=tuple((str(alias), str(target)) for alias, target in aliases.items()),
        focus_terms=tuple(str(term) for term in item.get("focus_terms", [])),
    )
