"""Deterministic evidence checks: a value survives only if the current document backs it.

The LLM proposes a value *and* a short literal quote for it. This module decides,
without any model involved, whether that pair is supported by the text fetched in
this request. Anything it cannot confirm becomes ``null``: a visible gap is
cheaper than a confident wrong number.

This module has no third-party imports so the runner can share it.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

FieldKind = Literal["number", "text"]
_NUMBER = re.compile(r"\d[\d.,]*")


@dataclass(frozen=True)
class FieldSpec:
    """One value to extract per record.

    ``keywords`` are words the source must mention for this field to be credible
    (for example ``("price", "$")``). They also steer which parts of a long page
    and which images reach the model.
    """

    name: str
    kind: FieldKind = "number"
    keywords: tuple[str, ...] = ()

    @property
    def evidence_key(self) -> str:
        return f"{self.name}_evidence"


def normalize_text(value: str) -> str:
    """Lowercase, strip accents and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.lower().split())


def numeric_token(token: str) -> str:
    """Normalize superficial formatting while preserving digit identity.

    ``"1,234.50"``, ``"1234.5"`` and ``"1234,5"`` all become ``"12345"``.
    """
    normalized = token.replace(",", "") if "." in token else token.replace(",", ".")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return re.sub(r"\D", "", normalized)


def numeric_variants(token: str) -> set[str]:
    """Every plausible reading of a number token.

    ``"1,000"`` is one thousand in most of the Americas and one point zero in much of
    Europe; both readings are kept so a correct value is never rejected on format alone.
    """
    variants = {numeric_token(token)}
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        variants.add(numeric_token(token.replace(",", "")))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        variants.add(numeric_token(token.replace(".", "").replace(",", ".")))
    return variants - {""}


def _mentions_field(field: FieldSpec, source_normalized: str) -> bool:
    if not field.keywords:
        return True
    compact_letters = re.sub(r"[^a-z]", "", source_normalized)
    for keyword in field.keywords:
        key = normalize_text(keyword)
        if not key:
            continue
        if key[0].isalnum() and key[-1].isalnum():
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", source_normalized):
                return True
        elif key in source_normalized:
            return True
        # PDFs often letter-space labels ("C A T"); compare letters only.
        letters = re.sub(r"[^a-z]", "", key)
        if len(letters) >= 3 and letters in compact_letters:
            return True
    return False


def is_supported(value: object, evidence: object, source_text: str, field: FieldSpec) -> bool:
    """True when ``value`` and its quoted ``evidence`` are both backed by ``source_text``.

    Numbers: every number in the value must appear in the quote and in the source.
    The quote itself does not have to appear verbatim, because OCR routinely splits
    a label and its number into different cells; the digits still must be there.

    Text: the value must be inside the quote and the quote inside the source.
    """
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    source_normalized = normalize_text(source_text)
    if not _mentions_field(field, source_normalized):
        return False
    if field.kind == "number":
        value_tokens = [numeric_variants(token) for token in _NUMBER.findall(str(value))]
        evidence_tokens = set().union(*(numeric_variants(t) for t in _NUMBER.findall(evidence)))
        source_tokens = set().union(*(numeric_variants(t) for t in _NUMBER.findall(source_text)))
        return bool(value_tokens) and all(
            variants & evidence_tokens and variants & source_tokens for variants in value_tokens
        )
    value_normalized = normalize_text(str(value))
    evidence_normalized = normalize_text(evidence)
    return (
        bool(value_normalized)
        and value_normalized in evidence_normalized
        and evidence_normalized in source_normalized
    )


def validate_records(data: dict[str, Any], source_text: str, fields: tuple[FieldSpec, ...]) -> int:
    """Null every unsupported value in place and return how many were rejected.

    Raises ``ValueError`` when the response shape itself is wrong.
    """
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("LLM response is missing records")
    rejected = 0
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("LLM response has an invalid record")
        for field in fields:
            value = record.get(field.name)
            if value is None:
                record[field.evidence_key] = None
                continue
            evidence = record.get(field.evidence_key)
            if not is_supported(value, evidence, source_text, field):
                logger.warning(
                    "unsupported value record=%s field=%s value=%r evidence=%r",
                    record.get("name"),
                    field.name,
                    value,
                    str(evidence)[:240] if evidence is not None else None,
                )
                record[field.name] = None
                record[field.evidence_key] = None
                rejected += 1
    return rejected
