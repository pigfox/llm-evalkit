"""Parsing never raises. Bad model output is a graded outcome."""

from __future__ import annotations

from evalkit.schemas import (
    Invoice,
    Measurement,
    Parsed,
    ParseFailure,
    Person,
    Sentiment,
    Topics,
    extract_json_block,
    parse_into,
)


def test_extract_json_block_prefers_a_fenced_block() -> None:
    text = 'Sure!\n```json\n{"label": "positive", "confidence": 0.9}\n```\nHope that helps.'
    assert extract_json_block(text) == '{"label": "positive", "confidence": 0.9}'


def test_extract_json_block_falls_back_to_outermost_braces() -> None:
    text = 'Here you go: {"label": "neutral", "confidence": 0.4} -- let me know.'
    assert extract_json_block(text) == '{"label": "neutral", "confidence": 0.4}'


def test_extract_json_block_returns_stripped_text_when_there_is_no_object() -> None:
    assert extract_json_block("  no object at all  ") == "no object at all"


def test_parse_into_returns_parsed_on_valid_output() -> None:
    result = parse_into(Sentiment, '{"label": "positive", "confidence": 0.9}')
    assert isinstance(result, Parsed)
    assert result.value.label == "positive"


def test_malformed_output_yields_parse_failure_not_an_exception() -> None:
    result = parse_into(Sentiment, "I think it is probably positive, honestly.")
    assert isinstance(result, ParseFailure)
    assert result.errors[0]["type"] == "json_decode"
    assert result.raw.startswith("I think")


def test_truncated_json_yields_parse_failure() -> None:
    result = parse_into(Sentiment, '{"label": "positive", "confid')
    assert isinstance(result, ParseFailure)
    assert result.errors[0]["type"] == "json_decode"


def test_a_json_array_is_a_parse_failure_with_a_named_reason() -> None:
    result = parse_into(Sentiment, "[1, 2, 3]")
    assert isinstance(result, ParseFailure)
    assert result.errors[0]["type"] == "not_an_object"
    assert "got list" in result.errors[0]["msg"]


def test_schema_violation_yields_parse_failure_carrying_pydantic_errors() -> None:
    result = parse_into(Sentiment, '{"label": "delighted", "confidence": 4.2}')
    assert isinstance(result, ParseFailure)
    kinds = {e["type"] for e in result.errors}
    assert len(result.errors) == 2
    assert any("literal" in k or "enum" in k for k in kinds)
    for err in result.errors:
        assert isinstance(err["loc"], list)
        assert isinstance(err["msg"], str)


def test_every_shipped_model_round_trips() -> None:
    assert isinstance(parse_into(Person, '{"name": "A", "age": 1}'), Parsed)
    assert isinstance(
        parse_into(Invoice, '{"total": 1.0, "currency": "USD", "line_items": []}'), Parsed
    )
    assert isinstance(parse_into(Topics, '{"primary": "x", "tags": []}'), Parsed)
    assert isinstance(parse_into(Measurement, '{"value": 1.0, "unit": "m"}'), Parsed)


def test_person_rejects_an_out_of_range_age() -> None:
    assert isinstance(parse_into(Person, '{"name": "A", "age": 900}'), ParseFailure)
