"""Pydantic models for the structured outputs under test, plus a parse result
union.

Parsing never raises on bad model output. A model that emits prose, truncated
JSON, or a well-formed object with the wrong field types produces a
``ParseFailure``, which is an ordinary graded outcome and is carried all the
way through to the report. Nothing in this package converts it to an exception.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, Field, ValidationError

M = TypeVar("M", bound=BaseModel)


class Sentiment(BaseModel):
    """A sentiment label with a bounded confidence."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


class Person(BaseModel):
    """A person extracted from free text."""

    name: str
    age: int = Field(ge=0, le=150)
    email: str | None = None


class Invoice(BaseModel):
    """An invoice total with a three-letter currency code."""

    total: float
    currency: str = Field(min_length=3, max_length=3)
    line_items: list[str]


class Topics(BaseModel):
    """A primary topic plus a tag set."""

    primary: str
    tags: list[str]


class Measurement(BaseModel):
    """A scalar measurement with its unit."""

    value: float
    unit: str


@dataclass(frozen=True)
class Parsed(Generic[M]):
    """Model output that validated against the case's schema."""

    value: M


@dataclass(frozen=True)
class ParseFailure:
    """Model output that did not validate. A graded outcome, not an error."""

    raw: str
    errors: list[dict[str, Any]]


ParseResult: TypeAlias = "Parsed[Any] | ParseFailure"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Pull the most plausible JSON object out of a model response.

    Models routinely wrap JSON in a markdown fence or in a sentence of
    preamble. Recovering from that is a property of the harness, not of the
    model, so it happens here and is applied identically to every case.
    """
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _clean_error(err: Any) -> dict[str, Any]:
    """Reduce a pydantic error to a JSON-serialisable record."""
    return {
        "type": str(err.get("type", "unknown")),
        "loc": [str(part) for part in err.get("loc", ())],
        "msg": str(err.get("msg", "")),
    }


def parse_into(model: type[M], text: str) -> ParseResult:
    """Parse ``text`` into ``model``, returning a result union rather than raising."""
    candidate = extract_json_block(text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseFailure(
            raw=text,
            errors=[{"type": "json_decode", "loc": [], "msg": str(exc)}],
        )
    if not isinstance(data, dict):
        return ParseFailure(
            raw=text,
            errors=[
                {
                    "type": "not_an_object",
                    "loc": [],
                    "msg": f"expected a JSON object, got {type(data).__name__}",
                }
            ],
        )
    try:
        value = model.model_validate(data)
    except ValidationError as exc:
        return ParseFailure(raw=text, errors=[_clean_error(e) for e in exc.errors()])
    return Parsed(value=value)
