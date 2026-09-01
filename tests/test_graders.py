"""Graders are deterministic and explain themselves on a pass as well as a fail."""

from __future__ import annotations

from typing import Any

from evalkit.graders import (
    Grade,
    GradeInput,
    exact_match,
    field_set_equal,
    numeric_within,
    schema_valid,
    tool_call_matches,
)
from evalkit.provider import RawResponse, ToolCall
from evalkit.schemas import ParseFailure, Sentiment, Topics, parse_into


def response(text: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> RawResponse:
    return RawResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason="end_turn",
        latency_ms=1,
        attempt_count=1,
        attempts=(),
    )


def item(text: str, *, model: Any = Sentiment, tool_calls: tuple[ToolCall, ...] = ()) -> GradeInput:
    return GradeInput(parsed=parse_into(model, text), response=response(text, tool_calls))


def unparsed(tool_calls: tuple[ToolCall, ...] = ()) -> GradeInput:
    return GradeInput(
        parsed=ParseFailure(raw="prose", errors=[{"type": "json_decode", "loc": [], "msg": "x"}]),
        response=response("prose", tool_calls),
    )


def test_grade_serialises() -> None:
    grade = Grade(passed=True, reason="because", observed=1, expected=1)
    assert grade.to_dict() == {"passed": True, "reason": "because", "observed": 1, "expected": 1}


def test_exact_match_passes_and_says_why() -> None:
    grade = exact_match("label", "positive")(item('{"label":"positive","confidence":0.9}'))
    assert grade.passed
    assert grade.reason == "label == 'positive'"


def test_exact_match_fails_and_reports_both_sides() -> None:
    grade = exact_match("label", "positive")(item('{"label":"negative","confidence":0.9}'))
    assert not grade.passed
    assert grade.observed == "negative"
    assert grade.expected == "positive"


def test_exact_match_on_unparsed_output_says_so() -> None:
    grade = exact_match("label", "positive")(unparsed())
    assert not grade.passed
    assert "did not parse" in grade.reason


def test_numeric_within_passes_inside_tolerance() -> None:
    grade = numeric_within("confidence", 0.9, 0.1)(item('{"label":"positive","confidence":0.85}'))
    assert grade.passed
    assert "within 0.1" in grade.reason


def test_numeric_within_fails_outside_tolerance_and_reports_the_delta() -> None:
    grade = numeric_within("confidence", 0.9, 0.01)(item('{"label":"positive","confidence":0.5}'))
    assert not grade.passed
    assert "delta 0.4" in grade.reason


def test_numeric_within_on_unparsed_output_says_so() -> None:
    grade = numeric_within("confidence", 0.9, 0.1)(unparsed())
    assert not grade.passed
    assert "did not parse" in grade.reason


def test_field_set_equal_ignores_order() -> None:
    grade = field_set_equal("tags", {"a", "b"})(
        item('{"primary":"x","tags":["b","a"]}', model=Topics)
    )
    assert grade.passed


def test_field_set_equal_names_what_is_missing_and_what_is_extra() -> None:
    grade = field_set_equal("tags", {"a", "b"})(
        item('{"primary":"x","tags":["a","c"]}', model=Topics)
    )
    assert not grade.passed
    assert "missing ['b']" in grade.reason
    assert "unexpected ['c']" in grade.reason


def test_field_set_equal_on_unparsed_output_says_so() -> None:
    grade = field_set_equal("tags", {"a"})(unparsed())
    assert not grade.passed
    assert "did not parse" in grade.reason


def test_schema_valid_passes_on_valid_output() -> None:
    grade = schema_valid()(item('{"label":"positive","confidence":0.9}'))
    assert grade.passed
    assert grade.observed["status"] == "parsed"


def test_schema_valid_fails_on_invalid_output_and_carries_the_errors() -> None:
    grade = schema_valid()(unparsed())
    assert not grade.passed
    assert grade.observed["status"] == "parse_failure"
    assert grade.observed["errors"][0]["type"] == "json_decode"


def test_schema_valid_expecting_failure_passes_when_output_does_not_parse() -> None:
    grade = schema_valid(expect_valid=False)(unparsed())
    assert grade.passed
    assert "what this case expects" in grade.reason
    assert grade.expected == "parse_failure"


def test_schema_valid_expecting_failure_fails_when_output_parses() -> None:
    grade = schema_valid(expect_valid=False)(item('{"label":"positive","confidence":0.9}'))
    assert not grade.passed
    assert "expects a parse failure" in grade.reason


def test_tool_call_matches_grades_on_name_and_arguments_and_ignores_prose() -> None:
    grader = tool_call_matches(
        "get_weather",
        lambda a: a.get("location") == "Reykjavik",
        predicate_description="location == 'Reykjavik'",
    )
    call = ToolCall(name="get_weather", arguments={"location": "Reykjavik", "unit": "celsius"})

    helpful = grader(unparsed(tool_calls=(call,)))
    misleading = grader(
        GradeInput(
            parsed=ParseFailure(raw="", errors=[]),
            response=response(
                "The weather in Paris is 40 degrees and I did not call any tool.", (call,)
            ),
        )
    )

    assert helpful.passed
    assert misleading.passed, "prose must not affect a tool-call grade"
    assert helpful.reason == misleading.reason


def test_tool_call_matches_fails_when_the_tool_was_never_called() -> None:
    grader = tool_call_matches("get_weather", lambda _a: True)
    grade = grader(unparsed(tool_calls=(ToolCall(name="get_time", arguments={}),)))
    assert not grade.passed
    assert "no tool call named 'get_weather'" in grade.reason
    assert grade.observed == {"tool_calls": ["get_time"]}


def test_tool_call_matches_fails_when_the_arguments_are_wrong() -> None:
    grader = tool_call_matches(
        "get_weather",
        lambda a: a.get("unit") == "celsius",
        predicate_description="unit == 'celsius'",
    )
    call = ToolCall(name="get_weather", arguments={"location": "Reykjavik", "unit": "fahrenheit"})
    grade = grader(unparsed(tool_calls=(call,)))
    assert not grade.passed
    assert "did not satisfy unit == 'celsius'" in grade.reason
    assert grade.observed["arguments"]["unit"] == "fahrenheit"


def test_prose_that_only_describes_a_call_does_not_pass() -> None:
    grader = tool_call_matches("get_weather", lambda _a: True)
    grade = grader(unparsed(tool_calls=()))
    assert not grade.passed
