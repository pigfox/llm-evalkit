"""The evaluation suite.

Cases live in code rather than in YAML for one reason: the lock hashes the
source text of this file and of ``graders.py`` together with the predictions,
so a case that is quietly reworded after pre-registration invalidates the lock.
A YAML suite would need a parallel hashing rule and would tempt everyone to
treat data edits as not-really-edits. They are edits.

Every case carries its prediction with it. ``predicted_pass`` is what we
committed to before running anything; ``predicted_notes`` says why. Two cases
are predicted to fail on purpose -- a suite in which every prediction is "pass"
is not a prediction, it is an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from evalkit.graders import (
    Grader,
    exact_match,
    field_set_equal,
    numeric_within,
    schema_valid,
    tool_call_matches,
)
from evalkit.schemas import Invoice, Measurement, Person, Sentiment, Topics

JSON_SYSTEM = "Reply with a single JSON object and nothing else."

WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "description": "Look up the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name."},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location", "unit"],
    },
}


@dataclass(frozen=True)
class Case:
    """One graded case, carrying its own pre-registered prediction."""

    id: str
    prompt: str
    schema: type[BaseModel] | None
    grader: Grader
    predicted_pass: bool
    predicted_notes: str
    system: str | None = JSON_SYSTEM
    tools: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _weather_args_ok(args: dict[str, Any]) -> bool:
    return args.get("location") == "Reykjavik" and args.get("unit") == "celsius"


SUITE: tuple[Case, ...] = (
    Case(
        id="sentiment_clear_positive",
        prompt=(
            "Classify the sentiment of this review and give a confidence between 0 and 1.\n"
            'Review: "Arrived a day early, works perfectly, and the price was fair."\n'
            "Fields: label (positive|negative|neutral), confidence."
        ),
        schema=Sentiment,
        grader=exact_match("label", "positive"),
        predicted_pass=True,
        predicted_notes="Unambiguously positive review with no hedging; this is the easy case.",
    ),
    Case(
        id="sentiment_ambiguous_hedge",
        prompt=(
            "Classify the sentiment of this review and give a confidence between 0 and 1.\n"
            'Review: "It does what it says. I have no strong feelings either way, though '
            'the packaging was excessive."\n'
            "Fields: label (positive|negative|neutral), confidence."
        ),
        schema=Sentiment,
        grader=exact_match("label", "neutral"),
        predicted_pass=False,
        predicted_notes=(
            "PREDICTED FAIL. The review is neutral overall but ends on a concrete complaint. "
            "Models weight the last clause heavily and we expect 'negative'."
        ),
    ),
    Case(
        id="sentiment_confidence_range",
        prompt=(
            "Classify the sentiment of this review and give a confidence between 0 and 1.\n"
            'Review: "Best purchase I have made all year. Everything about it is right."\n'
            "Fields: label (positive|negative|neutral), confidence."
        ),
        schema=Sentiment,
        grader=numeric_within("confidence", 0.9, 0.1),
        predicted_pass=True,
        predicted_notes="Emphatic review; self-reported confidence should land in [0.8, 1.0].",
    ),
    Case(
        id="person_name",
        prompt=(
            "Extract the person from this sentence as JSON with fields name, age, email.\n"
            '"Ada Lovelace, then 36, could be reached at ada@example.org."'
        ),
        schema=Person,
        grader=exact_match("name", "Ada Lovelace"),
        predicted_pass=True,
        predicted_notes="Name is stated verbatim; extraction is copying, not inference.",
    ),
    Case(
        id="person_age",
        prompt=(
            "Extract the person from this sentence as JSON with fields name, age, email.\n"
            '"Ada Lovelace, then 36, could be reached at ada@example.org."'
        ),
        schema=Person,
        grader=numeric_within("age", 36, 0),
        predicted_pass=True,
        predicted_notes="Age is stated as a numeral and needs no arithmetic.",
    ),
    Case(
        id="person_null_email",
        prompt=(
            "Extract the person from this sentence as JSON with fields name, age, email. "
            "Use null for anything the sentence does not state.\n"
            '"Grace Hopper was 45 at the time."'
        ),
        schema=Person,
        grader=exact_match("email", None),
        predicted_pass=True,
        predicted_notes=(
            "The instruction to use null is explicit; we expect null rather than an "
            "invented address or an empty string."
        ),
    ),
    Case(
        id="invoice_total",
        prompt=(
            "Read this invoice into JSON with fields total, currency, line_items.\n"
            "Consulting 1000.00 USD, Hosting 234.50 USD, Support 50.00 USD."
        ),
        schema=Invoice,
        grader=numeric_within("total", 1284.50, 0.01),
        predicted_pass=True,
        predicted_notes="Three-term addition; well inside what current models do reliably.",
    ),
    Case(
        id="invoice_currency",
        prompt=(
            "Read this invoice into JSON with fields total, currency, line_items.\n"
            "Consulting 1000.00 USD, Hosting 234.50 USD, Support 50.00 USD."
        ),
        schema=Invoice,
        grader=exact_match("currency", "USD"),
        predicted_pass=True,
        predicted_notes=(
            "Currency is stated three times and the schema constrains it to three characters."
        ),
    ),
    Case(
        id="invoice_line_items",
        prompt=(
            "Read this invoice into JSON with fields total, currency, line_items. "
            "line_items must list only the billed services.\n"
            "Consulting 1000.00 USD, Hosting 234.50 USD, Support 50.00 USD."
        ),
        schema=Invoice,
        grader=field_set_equal("line_items", {"Consulting", "Hosting", "Support"}),
        predicted_pass=True,
        predicted_notes=(
            "Three services are named explicitly and 'only the billed services' rules out "
            "a summary row."
        ),
    ),
    Case(
        id="topics_tags",
        prompt=(
            "Give the primary topic and up to four tags for this abstract, as JSON with "
            "fields primary and tags.\n"
            '"We present a cache eviction policy that reduces tail latency in distributed '
            'key-value stores under skewed read load."'
        ),
        schema=Topics,
        grader=field_set_equal("tags", {"caching", "latency", "distributed-systems"}),
        predicted_pass=True,
        predicted_notes=(
            "The three concepts are named in the abstract and the tag vocabulary is "
            "constrained by the prompt's 'up to four'."
        ),
    ),
    Case(
        id="refusal_is_not_json",
        prompt=(
            "In one short paragraph of ordinary prose, with no JSON and no code fence, "
            "explain why a cache eviction policy affects tail latency."
        ),
        schema=Topics,
        grader=schema_valid(expect_valid=False),
        predicted_pass=True,
        predicted_notes=(
            "The prompt forbids JSON, so the correct outcome is a parse failure. This case "
            "exists to keep ParseFailure a graded result rather than an unhandled hole."
        ),
        system=None,
    ),
    Case(
        id="measurement_valid_schema",
        prompt=(
            "Convert this to JSON with fields value and unit.\n"
            "The probe recorded three hundred and forty-three metres per second."
        ),
        schema=Measurement,
        grader=schema_valid(),
        predicted_pass=True,
        predicted_notes="Two flat fields with no enum; the schema is easy to satisfy.",
    ),
    Case(
        id="measurement_unit_literal",
        prompt=(
            "Convert this to JSON with fields value and unit.\n"
            "The probe recorded three hundred and forty-three metres per second."
        ),
        schema=Measurement,
        grader=exact_match("unit", "m/s"),
        predicted_pass=False,
        predicted_notes=(
            "PREDICTED FAIL. The prompt spells the unit out in words and never asks for "
            "symbol form, so we expect 'meters per second' back. The grader demands 'm/s'. "
            "This is a badly specified case and it is kept to show the harness reporting one."
        ),
    ),
    Case(
        id="weather_tool_call",
        prompt="What is the temperature in Reykjavik right now? Answer in celsius.",
        schema=None,
        grader=tool_call_matches(
            "get_weather",
            _weather_args_ok,
            predicate_description="location == 'Reykjavik' and unit == 'celsius'",
        ),
        predicted_pass=True,
        predicted_notes=(
            "Both arguments are stated in the prompt and the tool has a required enum. "
            "Graded on the call alone; whatever prose accompanies it is ignored."
        ),
        system=None,
        tools=(WEATHER_TOOL,),
    ),
    Case(
        id="sentiment_retry_then_ok",
        prompt=(
            "Classify the sentiment of this review and give a confidence between 0 and 1.\n"
            'Review: "Broke within a week and support never replied."\n'
            "Fields: label (positive|negative|neutral), confidence."
        ),
        schema=Sentiment,
        grader=exact_match("label", "negative"),
        predicted_pass=True,
        predicted_notes=(
            "Unambiguously negative. The fixture scripts a 529 on the first attempt so the "
            "default suite exercises the retry path and the attempt records are non-empty."
        ),
    ),
    Case(
        id="sentiment_unstable",
        prompt=(
            "Classify the sentiment of this review and give a confidence between 0 and 1.\n"
            'Review: "Cheap, and it shows -- but for the money I cannot really complain."\n'
            "Fields: label (positive|negative|neutral), confidence."
        ),
        schema=Sentiment,
        grader=exact_match("label", "positive"),
        predicted_pass=True,
        predicted_notes=(
            "A backhanded compliment that resolves positive on balance. We predict the "
            "model agrees, and predict it does so consistently."
        ),
    ),
)


def by_id(case_id: str) -> Case:
    """Look one case up, raising a useful error rather than an IndexError."""
    for case in SUITE:
        if case.id == case_id:
            return case
    raise KeyError(f"no case with id {case_id!r}")
