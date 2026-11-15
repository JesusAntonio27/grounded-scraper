"""CLI used by the Airflow DAG: extract one isolated source, or build the run's rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from runner.client import ApiRequestError, ScraperApiClient, records_from_response
from runner.rows import build_rows
from runner.snapshot import normalize_period, normalize_run_id, write_source_result
from runner.sources import Catalog, load_catalog

logger = logging.getLogger("grounded_runner")


def _local_captured_at() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fallback_diagnostic(reason: str) -> str:
    concise = " ".join(reason.split())[:180]
    return f"The source could not be processed: {concise or 'technical failure without detail.'}"


def failure_category(reason: str) -> str:
    value = reason.lower()
    if "missing records" in value or "without current evidence" in value:
        return "page-changed"
    if "timeout" in value or "timed out" in value or "connection" in value or "http" in value:
        return "upstream-unavailable"
    if "not found" in value:
        return "not-found"
    if "invalid" in value or "incomplete" in value:
        return "extraction-failed"
    return "uncertain"


def run_source(
    catalog: Catalog,
    source_id: str,
    period: str,
    run_id: str,
    data_dir: str | Path,
    *,
    client: ScraperApiClient | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract one source and always leave an artifact for the load step.

    A source succeeds only when every requested record came back with at least one
    verified value; otherwise the next mode is tried. A source-level failure is a
    normal, auditable result: it is written down, not raised.
    """
    period = normalize_period(period)
    run_id = normalize_run_id(run_id)
    source = catalog.get_source(source_id)
    fields = catalog.fields
    if dry_run:
        return {
            "source_id": source.source_id,
            "source": source.url,
            "period": period,
            "run_id": run_id,
            "status": "dry_run",
            "records": list(source.records),
        }

    captured_at = _local_captured_at()
    client = client or ScraperApiClient.from_environment()
    attempt_errors: list[str] = []
    excerpt = ""
    partial_output: dict[str, Any] | None = None
    for mode in source.modes:
        try:
            response = client.extract(source, fields, mode=mode)
        except ApiRequestError as exc:
            attempt_errors.append(f"{mode}: {exc}")
            logger.warning("source=%s mode=%s status=failed error=%s", source_id, mode, exc)
            continue
        meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        excerpt = str(
            response.get("diagnostic_excerpt") or meta.get("diagnostic_excerpt") or excerpt
        )
        if isinstance(response.get("partial_output"), dict):
            partial_output = response["partial_output"]
        if response.get("status") != "extracted":
            attempt_errors.append(f"{mode}: {response.get('error') or 'extraction failed'}")
            continue

        records = records_from_response(response, source, fields)
        missing = sorted(set(source.records) - {record["name"] for record in records})
        empty = [r["name"] for r in records if all(r.get(spec.name) is None for spec in fields)]
        if missing or empty:
            detail = [f"missing records ({', '.join(missing)})"] if missing else []
            detail += [f"{name} without current evidence" for name in empty]
            attempt_errors.append(f"{mode}: {'; '.join(detail)}")
            partial_output = {"records": records}
            logger.warning("source=%s mode=%s status=incomplete detail=%s", source_id, mode, detail)
            continue

        partial_fields = [
            {"name": r["name"], "missing_fields": [s.name for s in fields if r.get(s.name) is None]}
            for r in records
            if any(r.get(spec.name) is None for spec in fields)
        ]
        diagnostic = origin = None
        if partial_fields:
            reason = "Fields without current, verifiable evidence: " + "; ".join(
                f"{item['name']}: {', '.join(item['missing_fields'])}" for item in partial_fields
            )
            llm_diagnostic = client.diagnose(
                source.url, reason, current_excerpt=excerpt, partial_output={"records": records}
            )
            diagnostic = llm_diagnostic or fallback_diagnostic(reason)
            origin = "llm" if llm_diagnostic else "fallback"
        missing_by_name = {item["name"]: item["missing_fields"] for item in partial_fields}
        output_records = []
        for record in records:
            item = {**record, "source": source.url, "captured_at": captured_at}
            if record["name"] in missing_by_name:
                item.update(
                    missing_fields=missing_by_name[record["name"]],
                    diagnostic=diagnostic,
                    diagnostic_origin=origin,
                )
            output_records.append(item)
        result = {
            "source_id": source.source_id,
            "source": source.url,
            "period": period,
            "run_id": run_id,
            "status": "success",
            "captured_at": captured_at,
            "mode": mode,
            "records": output_records,
            "partial_fields": partial_fields,
            "diagnostic": diagnostic,
            "diagnostic_origin": origin,
            "api_meta": meta,
        }
        write_source_result(data_dir, period, run_id, source.source_id, result)
        logger.info("source=%s status=success mode=%s records=%d", source_id, mode, len(records))
        return result

    reason = " | ".join(attempt_errors) or "No extraction mode is configured for this source."
    llm_diagnostic = client.diagnose(
        source.url, reason, current_excerpt=excerpt, partial_output=partial_output
    )
    result = {
        "source_id": source.source_id,
        "source": source.url,
        "period": period,
        "run_id": run_id,
        "status": "failed",
        "captured_at": captured_at,
        "records": [],
        "diagnostic": llm_diagnostic or fallback_diagnostic(reason),
        "diagnostic_origin": "llm" if llm_diagnostic else "fallback",
        "error": reason[:500],
        "failure_category": failure_category(reason),
        "modes_attempted": list(source.modes),
    }
    write_source_result(data_dir, period, run_id, source.source_id, result)
    logger.error("source=%s status=failed category=%s", source_id, result["failure_category"])
    return result


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="grounded-scraper runner")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--sources-file",
        default=os.getenv("GROUNDED_SOURCES_FILE", "sources.json"),
        help="catalog JSON (default: $GROUNDED_SOURCES_FILE)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run-source", help="extract one isolated source")
    run.add_argument("--source-id", required=True)
    run.add_argument("--period", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--dry-run", action="store_true")

    rows = commands.add_parser("build-rows", help="build database rows for exactly one run")
    rows.add_argument("--period", required=True)
    rows.add_argument("--run-id", required=True)
    rows.add_argument("--data-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    catalog = load_catalog(args.sources_file)
    if args.command == "run-source":
        result = run_source(
            catalog, args.source_id, args.period, args.run_id, args.data_dir, dry_run=args.dry_run
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        # Exit 0 even for a failed source: its artifact is already written and the
        # load step turns it into an "unavailable" row. Unexpected exceptions still
        # exit non-zero, which makes Airflow retry the task.
        return 0
    rows = build_rows(catalog, args.data_dir, args.period, args.run_id)
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
