"""Idempotent audit artifacts: one JSON file per (period, run, source), written atomically."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from runner.sources import SOURCE_ID_PATTERN, SourceSpec


def normalize_period(value: str) -> str:
    """Accept only the first day of a month: the business key of a snapshot."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("period must use YYYY-MM-DD") from exc
    if parsed.day != 1:
        raise ValueError("period must be the first day of its month")
    return parsed.isoformat()


def normalize_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value or ""):
        raise ValueError("run_id must contain only letters, digits, _, . or -")
    return value


def run_directory(data_dir: str | Path, period: str, run_id: str) -> Path:
    return (
        Path(data_dir).expanduser().resolve()
        / "runs"
        / normalize_period(period)
        / normalize_run_id(run_id)
    )


def source_result_path(data_dir: str | Path, period: str, run_id: str, source_id: str) -> Path:
    if not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError(f"invalid source_id {source_id!r}")
    return run_directory(data_dir, period, run_id) / "sources" / f"{source_id}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_name = handle.name
    os.replace(temporary_name, path)


def write_source_result(
    data_dir: str | Path, period: str, run_id: str, source_id: str, result: dict[str, Any]
) -> Path:
    """Overwrite one source's result for a run; retries never append."""
    path = source_result_path(data_dir, period, run_id, source_id)
    write_json_atomic(path, result)
    return path


def load_source_result(
    data_dir: str | Path, period: str, run_id: str, source: SourceSpec
) -> dict[str, Any] | None:
    """Read a source artifact, or a failed placeholder if it is unreadable or foreign."""
    path = source_result_path(data_dir, period, run_id, source.source_id)
    if not path.exists():
        return None

    def failed(diagnostic: str, error: str) -> dict[str, Any]:
        return {
            "source_id": source.source_id,
            "source": source.url,
            "period": normalize_period(period),
            "status": "failed",
            "diagnostic": diagnostic,
            "error": error,
            "records": [],
        }

    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return failed(
            "The source artifact could not be read.",
            f"Invalid source artifact: {type(exc).__name__}",
        )
    if (
        not isinstance(result, dict)
        or result.get("source_id") != source.source_id
        or result.get("period") != normalize_period(period)
        or result.get("run_id") != normalize_run_id(run_id)
    ):
        return failed(
            "The artifact does not belong to this source and run.",
            "Source artifact identity mismatch",
        )
    return result
