"""Internal HTTP surface consumed by the runner (and therefore by Airflow)."""

from __future__ import annotations

import logging
import secrets
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import get_settings
from app.evidence import FieldSpec
from app.extraction import (
    ExtractionError,
    ExtractionTask,
    diagnose_failure,
    extract_html,
    extract_pdf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="grounded-scraper", version="1.0.0", docs_url=None, redoc_url=None)


def _http_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value


class FieldModel(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    kind: Literal["number", "text"] = "number"
    keywords: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value == "name" or value.endswith("_evidence"):
            raise ValueError("field name is reserved")
        return value

    def to_spec(self) -> FieldSpec:
        return FieldSpec(self.name, self.kind, tuple(k for k in self.keywords if k.strip()))


class FetchOptions(BaseModel):
    render_js: bool = True
    geo_location: str | None = None
    timeout: int = Field(default=120, ge=5, le=300)
    user_agent_type: str = "desktop"
    browser_instructions: list[dict[str, Any]] = Field(default_factory=list)


class LLMOptions(BaseModel):
    model: str | None = None
    mode: Literal["text", "multimodal"] = "text"


class _TaskModel(BaseModel):
    records: list[str] = Field(min_length=1, max_length=50)
    fields: list[FieldModel] = Field(min_length=1, max_length=20)
    instructions: str = Field(default="", max_length=4_000)
    focus_terms: list[str] = Field(default_factory=list, max_length=30)
    llm_options: LLMOptions = Field(default_factory=LLMOptions)

    @model_validator(mode="after")
    def _unique_fields(self) -> _TaskModel:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        return self

    def task(self, url: str) -> ExtractionTask:
        return ExtractionTask(
            url=url,
            records=tuple(self.records),
            fields=tuple(field.to_spec() for field in self.fields),
            instructions=self.instructions,
            focus_terms=tuple(self.focus_terms),
        )


class ExtractRequest(_TaskModel):
    url: str
    fetch_options: FetchOptions = Field(default_factory=FetchOptions)

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return _http_url(value)


class PdfExtractRequest(_TaskModel):
    pdf_url: str

    @field_validator("pdf_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return _http_url(value)


class DiagnoseRequest(BaseModel):
    source_url: str
    failure_reason: str = Field(max_length=500)
    current_excerpt: str = Field(default="", max_length=1_200)
    partial_output: dict[str, Any] | None = None


async def _authorize(x_llm_scraper_token: str | None = Header(default=None)) -> None:
    configured = get_settings().internal_token
    if configured and not (
        x_llm_scraper_token and secrets.compare_digest(configured, x_llm_scraper_token)
    ):
        raise HTTPException(status_code=401, detail="invalid internal token")


@app.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    ready = settings.fetch_configured and settings.llm_configured
    return {
        # Deliberately no upstream call: checking quota would itself be billable.
        "status": "ready" if ready else "degraded",
        "fetch_backend": settings.fetch_backend,
        "fetch": "configured" if settings.fetch_configured else "missing_credentials",
        "llm": "configured" if settings.llm_configured else "missing_credentials",
    }


@app.post("/extract", dependencies=[Depends(_authorize)])
async def extract(request: ExtractRequest) -> dict[str, Any]:
    try:
        result = await extract_html(
            request.task(request.url),
            fetch_options=request.fetch_options.model_dump(),
            llm_options=request.llm_options.model_dump(),
        )
    except ExtractionError as exc:
        logger.warning("extract failed url=%s reason=%s", request.url, exc)
        return _failed(request.url, exc)
    return {"url": request.url, **result}


@app.post("/extract/pdf", dependencies=[Depends(_authorize)])
async def extract_pdf_endpoint(request: PdfExtractRequest) -> dict[str, Any]:
    try:
        result = await extract_pdf(
            request.task(request.pdf_url), llm_options=request.llm_options.model_dump()
        )
    except ExtractionError as exc:
        logger.warning("extract pdf failed url=%s reason=%s", request.pdf_url, exc)
        return _failed(request.pdf_url, exc)
    return {"url": request.pdf_url, **result}


@app.post("/diagnose", dependencies=[Depends(_authorize)])
async def diagnose(request: DiagnoseRequest) -> dict[str, str]:
    return {
        "diagnostic": await diagnose_failure(
            request.source_url,
            request.failure_reason,
            request.current_excerpt,
            request.partial_output,
        )
    }


def _failed(url: str, exc: ExtractionError) -> dict[str, Any]:
    # A source failure is a normal, auditable outcome, so it is a 200 with
    # status "failed" rather than an HTTP error the caller would retry blindly.
    return {
        "url": url,
        "status": "failed",
        "records": [],
        "error": str(exc),
        "diagnostic_excerpt": exc.current_excerpt,
        "partial_output": exc.partial_output,
    }
