"""Turn a run's source artifacts into one database row per (source, record, period).

Failed or missing sources still produce rows, with null values and a status, so a
gap in the data is visible instead of silently missing.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from runner.snapshot import load_source_result, normalize_period, normalize_run_id
from runner.sources import Catalog

_NUMBER = re.compile(
    r"[$€£]?\s*(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)\s*%?",
)
_RANGE = re.compile(
    r"(?:(?:from|desde)\s+)?[$€£]?\s*(?P<minimum>\d+(?:[.,]\d+)?)\s*%?\s*"
    r"(?:(?:and\s+|y\s+)?(?:to|up to|hasta)|a|[-–—])\s*"
    r"[$€£]?\s*(?P<maximum>\d+(?:[.,]\d+)?)\s*%?",
    re.IGNORECASE,
)
_PAGE_CHANGE_MARKERS = (
    "missing records",
    "without current evidence",
    "invalid source artifact",
    "identity mismatch",
)


def capture_timezone() -> ZoneInfo:
    return ZoneInfo(os.getenv("CAPTURE_TIMEZONE", "UTC"))


def parse_decimal(value: object) -> Decimal | None:
    """Parse one literal number such as ``12.5``, ``"95.2%"`` or ``"$1,299"``.

    Ranges, formulas and free text stay ``None``: they keep their raw value instead
    of being collapsed into a misleading scalar.
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    match = _NUMBER.fullmatch(text)
    if match is None:
        return None
    numeric = match.group("number")
    if "," in numeric and "." in numeric or re.fullmatch(r"\d{1,3}(?:,\d{3})+", numeric):
        numeric = numeric.replace(",", "")
    elif "," in numeric:
        numeric = numeric.replace(",", ".")
    try:
        return Decimal(numeric)
    except InvalidOperation:
        return None


def parse_range(value: object) -> tuple[Decimal, Decimal] | None:
    """Parse an explicit literal range (``"from 30% to 90%"``, ``"30-90"``)."""
    if value is None:
        return None
    match = _RANGE.search(str(value))
    if match is None:
        return None
    try:
        minimum = Decimal(match.group("minimum").replace(",", "."))
        maximum = Decimal(match.group("maximum").replace(",", "."))
    except InvalidOperation:
        return None
    return (minimum, maximum) if minimum <= maximum else None


def _raw(value: object) -> str | None:
    return None if value is None else (str(value).strip() or None)


def _captured_at(value: object) -> datetime:
    zone = capture_timezone()
    if value is None:
        return datetime.now(zone)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        # One malformed artifact must degrade its own row, not abort the whole load.
        return datetime.now(zone)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def failure_status(error: object) -> str:
    normalized = str(error or "").lower()
    return "page-changed" if any(m in normalized for m in _PAGE_CHANGE_MARKERS) else "unavailable"


def build_rows(
    catalog: Catalog, data_dir: str | Path, period: str, run_id: str
) -> list[dict[str, Any]]:
    period = normalize_period(period)
    run_id = normalize_run_id(run_id)
    rows: list[dict[str, Any]] = []
    for source in catalog.sources:
        result = load_source_result(data_dir, period, run_id, source) or {}
        succeeded = result.get("status") == "success"
        by_name = {
            str(item.get("name")): item
            for item in result.get("records") or []
            if isinstance(item, dict) and item.get("name") in source.records
        }
        failure_reason = result.get("error") or result.get("diagnostic")
        for record_name in source.records:
            record = by_name.get(record_name) if succeeded else None
            raw_values = {
                spec.name: _raw(record.get(spec.name)) if record else None
                for spec in catalog.fields
            }
            field_values: dict[str, Any] = {}
            field_ranges: dict[str, list[Decimal]] = {}
            for spec in catalog.fields:
                raw = raw_values[spec.name]
                if spec.kind == "number":
                    field_values[spec.name] = parse_decimal(raw)
                    bounds = parse_range(raw) if field_values[spec.name] is None else None
                    if bounds:
                        field_ranges[spec.name] = list(bounds)
                else:
                    field_values[spec.name] = raw
            missing = [name for name, raw in raw_values.items() if raw is None]
            captured = len(missing) < len(raw_values)
            if captured and missing:
                diagnostic = (record or {}).get("diagnostic") or (
                    "Partial capture: no current evidence for " + ", ".join(missing) + "."
                )
            elif captured:
                diagnostic = None
            else:
                diagnostic = result.get("diagnostic") or "No current, verifiable value."
            rows.append(
                {
                    "source_id": source.source_id,
                    "record_name": record_name,
                    "source_url": source.url,
                    "snapshot_period": period,
                    "field_values": field_values,
                    "raw_field_values": raw_values,
                    "field_ranges": field_ranges,
                    "status": "captured" if captured else failure_status(failure_reason),
                    "diagnostic": diagnostic,
                    "captured_at": _captured_at(
                        (record or {}).get("captured_at") or result.get("captured_at")
                    ),
                    "raw_capture": result or None,
                }
            )
    return rows
