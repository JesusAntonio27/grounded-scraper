"""Monthly Airflow DAG for grounded-scraper.

Flow: start a short-lived extractor container -> health check -> one task per
catalog source (dynamic task mapping, so one broken page never blocks the rest)
-> build rows -> insert into PostgreSQL -> collect logs -> stop the container.

Credentials are read from the worker environment and handed to the container by
name (``docker run -e NAME``), so their values never appear in an argv or a log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from datetime import timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pendulum
import psycopg2
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.operators.python import get_current_context
from airflow.utils.trigger_rule import TriggerRule
from psycopg2.extras import Json, execute_values

LOGGER = logging.getLogger(__name__)

DAG_ID = "grounded_scraper_monthly"
IMAGE = os.getenv("GROUNDED_SCRAPER_IMAGE", "grounded-scraper:latest")
SOURCES_FILE = os.getenv("GROUNDED_SOURCES_FILE", "/opt/airflow/config/grounded_sources.json")
DB_CONN_ID = os.getenv("GROUNDED_SCRAPER_DB_CONN_ID", "grounded_scraper_db")
TABLE = os.getenv("GROUNDED_SCRAPER_TABLE", "public.grounded_snapshots")
EXPECTED_DATABASE = os.getenv("GROUNDED_SCRAPER_EXPECTED_DB") or None
SCHEDULE_TZ = pendulum.timezone(os.getenv("GROUNDED_SCRAPER_TZ", "UTC"))
CONTAINER_PREFIX = "grounded-scraper"
CONTAINER_DATA_DIR = "/tmp/grounded"
CONTAINER_SOURCES_FILE = f"{CONTAINER_DATA_DIR}/sources.json"
PASSED_ENV_NAMES = (
    "FETCH_BACKEND",
    "OXYLABS_USER",
    "OXYLABS_PASS",
    "OXYLABS_GEO_LOCATION",
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_SCRAPER_INTERNAL_TOKEN",
    "OCR_LANGUAGES",
    "CAPTURE_TIMEZONE",
)
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _period(context: dict[str, Any]) -> str:
    """Business period: ``dag_run.conf["period"]`` for backfills, else the interval's month."""
    conf = getattr(context.get("dag_run"), "conf", None) or {}
    requested = conf.get("period")
    if requested is not None:
        try:
            parsed = pendulum.parse(str(requested), strict=True).date()
        except (TypeError, ValueError) as exc:
            raise AirflowException("dag_run.conf.period must be YYYY-MM-01") from exc
        if parsed.day != 1 or str(requested) != parsed.to_date_string():
            raise AirflowException("dag_run.conf.period must be the first day: YYYY-MM-01")
        return parsed.to_date_string()
    start = context.get("data_interval_start")
    if start is None:
        raise AirflowException("Airflow did not provide data_interval_start")
    return start.in_timezone(SCHEDULE_TZ).start_of("month").to_date_string()


def _run_id(context: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(getattr(context.get("dag_run"), "run_id", "")))
    if not safe:
        raise AirflowException("Airflow did not provide a valid run_id")
    return safe[:160]


def _container_name(context: dict[str, Any]) -> str:
    digest = hashlib.sha256(_run_id(context).encode("utf-8")).hexdigest()[:12]
    return f"{CONTAINER_PREFIX}-{digest}"


def _docker(
    *args: str, timeout: int = 120, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides or {})
    try:
        return subprocess.run(
            ["docker", *args], text=True, capture_output=True, check=False, timeout=timeout, env=env
        )
    except FileNotFoundError as exc:
        raise AirflowException("The Docker CLI is not available on this worker") from exc
    except subprocess.TimeoutExpired as exc:
        raise AirflowException(f"docker {args[0]} exceeded {timeout}s") from exc


def _docker_network() -> str:
    """The worker's own Docker network, so the worker can reach the container by name."""
    configured = os.getenv("GROUNDED_SCRAPER_DOCKER_NETWORK")
    if configured:
        return configured
    hostname = os.getenv("HOSTNAME")
    if not hostname:
        raise AirflowException("Set GROUNDED_SCRAPER_DOCKER_NETWORK")
    inspected = _docker("inspect", "--format", "{{json .NetworkSettings.Networks}}", hostname)
    if inspected.returncode:
        raise AirflowException(
            "Cannot inspect the worker network; set GROUNDED_SCRAPER_DOCKER_NETWORK"
        )
    networks = json.loads(inspected.stdout or "{}")
    candidates = [name for name in networks if name not in {"bridge", "host", "none"}]
    if len(candidates) != 1:
        raise AirflowException(
            f"Ambiguous worker network {candidates}; set GROUNDED_SCRAPER_DOCKER_NETWORK"
        )
    return candidates[0]


def _runner(command: str, context: dict[str, Any], source_id: str | None = None) -> str:
    args = [
        "exec",
        _container_name(context),
        "python",
        "-m",
        "runner.cli",
        "--sources-file",
        CONTAINER_SOURCES_FILE,
        command,
    ]
    if source_id is not None:
        args += ["--source-id", source_id]
    args += [
        "--period",
        _period(context),
        "--run-id",
        _run_id(context),
        "--data-dir",
        CONTAINER_DATA_DIR,
    ]
    completed = _docker(*args, timeout=int(timedelta(minutes=20).total_seconds()))
    if completed.stderr:
        LOGGER.info("runner stderr (%s):\n%s", source_id or command, completed.stderr.rstrip())
    if completed.returncode:
        raise AirflowException(
            f"runner exited with {completed.returncode} for {source_id or command}"
        )
    return completed.stdout.strip()


def _table() -> str:
    if not _TABLE_NAME.fullmatch(TABLE):
        raise AirflowException("GROUNDED_SCRAPER_TABLE must be [schema.]table")
    return TABLE


@dag(
    dag_id=DAG_ID,
    description="Monthly grounded extraction, one isolated task per source",
    start_date=pendulum.datetime(2026, 1, 1, tz=SCHEDULE_TZ),
    schedule="0 10 1 * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
    },
    tags=["scraping", "llm"],
)
def grounded_scraper() -> None:
    @task(execution_timeout=timedelta(minutes=3))
    def start_api() -> None:
        context = get_current_context()
        _table()
        name = _container_name(context)
        stale = _docker("ps", "-aq", "--filter", f"name=^{name}$")
        if stale.stdout.strip():
            _docker("rm", "-f", name, timeout=45)
        passed = {key: os.environ[key] for key in PASSED_ENV_NAMES if os.environ.get(key)}
        args = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            _docker_network(),
            "--label",
            f"airflow.dag_id={DAG_ID}",
        ]
        for key in passed:
            args += ["-e", key]  # value comes from the subprocess env, not the argv
        created = _docker(*args, IMAGE, env_overrides=passed)
        if created.returncode:
            raise AirflowException(f"could not start the extractor: {created.stderr.strip()[:300]}")
        if _docker("exec", name, "mkdir", "-p", CONTAINER_DATA_DIR).returncode:
            raise AirflowException("could not prepare the container data directory")
        copied = _docker("cp", SOURCES_FILE, f"{name}:{CONTAINER_SOURCES_FILE}")
        if copied.returncode:
            raise AirflowException(f"could not copy the catalog: {copied.stderr.strip()[:300]}")

    @task(execution_timeout=timedelta(minutes=2))
    def preflight() -> None:
        url = f"http://{_container_name(get_current_context())}:8000/health"
        deadline, last = time.monotonic() + 60, "no response"
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=5) as response:  # noqa: S310 - internal container URL
                    status = str(json.loads(response.read().decode("utf-8")).get("status"))
                if status == "ready":
                    return
                if status == "degraded":
                    raise AirflowException("extractor started without fetch or LLM credentials")
                last = f"status {status}"
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = type(exc).__name__
            time.sleep(2)
        raise AirflowException(f"extractor not ready after 60s: {last}")

    @task
    def list_sources() -> list[str]:
        with open(SOURCES_FILE, encoding="utf-8") as handle:
            return [source["source_id"] for source in json.load(handle)["sources"]]

    @task(execution_timeout=timedelta(minutes=20))
    def run_source(source_id: str) -> None:
        _runner("run-source", get_current_context(), source_id=source_id)

    @task(trigger_rule=TriggerRule.ALL_DONE, execution_timeout=timedelta(minutes=5))
    def load_snapshot() -> None:
        context = get_current_context()
        rows = json.loads(_runner("build-rows", context))
        if not rows:
            LOGGER.warning(
                "period=%s: the catalog produced no rows; nothing to load", _period(context)
            )
            return
        values = [
            (
                row["source_id"],
                row["record_name"],
                row["source_url"],
                row["snapshot_period"],
                Json(row["field_values"]),
                Json(row["raw_field_values"]),
                Json(row["field_ranges"]),
                row["status"],
                row["diagnostic"],
                row["captured_at"],
                Json(row["raw_capture"]),
            )
            for row in rows
        ]
        sql = f"""
            INSERT INTO {_table()} (
                source_id, record_name, source_url, snapshot_period,
                field_values, raw_field_values, field_ranges,
                status, diagnostic, captured_at, raw_capture
            ) VALUES %s
            ON CONFLICT (source_id, record_name, snapshot_period) DO NOTHING
            RETURNING id
        """
        connection = psycopg2.connect(BaseHook.get_connection(DB_CONN_ID).get_uri())
        try:
            with connection.cursor() as cursor:
                if EXPECTED_DATABASE:
                    cursor.execute("SELECT current_database()")
                    if cursor.fetchone()[0] != EXPECTED_DATABASE:
                        raise AirflowException("connection resolved to an unexpected database")
                inserted = execute_values(cursor, sql, values, fetch=True) or []
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        LOGGER.info("period=%s rows=%d inserted=%d", _period(context), len(rows), len(inserted))

    @task(trigger_rule=TriggerRule.ALL_DONE, retries=0)
    def collect_logs() -> None:
        logs = _docker("logs", "--timestamps", _container_name(get_current_context()))
        LOGGER.info("extractor logs:\n%s%s", logs.stdout.rstrip(), logs.stderr.rstrip())

    @task(trigger_rule=TriggerRule.ALL_DONE, retries=0, execution_timeout=timedelta(minutes=1))
    def stop_api() -> None:
        name = _container_name(get_current_context())
        _docker("stop", "-t", "30", name, timeout=45)
        _docker("rm", name)

    started = start_api()
    ready = preflight()
    extracted = run_source.expand(source_id=list_sources())
    loaded = load_snapshot()
    started >> ready >> extracted >> loaded >> collect_logs() >> stop_api()


grounded_scraper()
