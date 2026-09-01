"""Structural guarantees about the suite itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.cases import SUITE, WEATHER_TOOL, _weather_args_ok, by_id
from evalkit.graders import GradeInput
from evalkit.provider import RawResponse
from evalkit.schemas import ParseFailure

STUB_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "stub"


def test_the_suite_has_at_least_twelve_cases() -> None:
    assert len(SUITE) >= 12


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in SUITE]
    assert len(ids) == len(set(ids))


def test_at_least_two_cases_are_predicted_to_fail_with_a_stated_reason() -> None:
    predicted_fail = [c for c in SUITE if not c.predicted_pass]
    assert len(predicted_fail) >= 2
    for case in predicted_fail:
        assert "PREDICTED FAIL" in case.predicted_notes


def test_every_case_registers_a_reason() -> None:
    for case in SUITE:
        assert case.predicted_notes.strip()


def test_exactly_one_case_calls_a_tool() -> None:
    tool_cases = [c for c in SUITE if c.tools]
    assert len(tool_cases) == 1
    assert tool_cases[0].id == "weather_tool_call"
    assert tool_cases[0].tools == (WEATHER_TOOL,)


def test_the_tool_case_is_graded_without_a_schema() -> None:
    case = by_id("weather_tool_call")
    assert case.schema is None


def test_at_least_one_case_expects_a_parse_failure() -> None:
    case = by_id("refusal_is_not_json")
    empty = RawResponse(
        text="", tool_calls=(), stop_reason=None, latency_ms=0, attempt_count=1, attempts=()
    )
    grade = case.grader(GradeInput(parsed=ParseFailure(raw="prose", errors=[]), response=empty))
    assert grade.passed, "the parse-failure case must pass when the output does not parse"


def test_every_case_has_a_stub_fixture() -> None:
    for case in SUITE:
        assert (STUB_FIXTURES / f"{case.id}.json").exists(), case.id


def test_no_fixture_is_orphaned() -> None:
    ids = {c.id for c in SUITE}
    for path in STUB_FIXTURES.glob("*.json"):
        assert path.stem in ids, path


def test_by_id_finds_a_case() -> None:
    assert by_id("person_name").id == "person_name"


def test_by_id_raises_a_named_error_for_an_unknown_id() -> None:
    with pytest.raises(KeyError, match="no case with id 'nope'"):
        by_id("nope")


def test_the_weather_predicate_requires_both_arguments() -> None:
    assert _weather_args_ok({"location": "Reykjavik", "unit": "celsius"})
    assert not _weather_args_ok({"location": "Reykjavik", "unit": "fahrenheit"})
    assert not _weather_args_ok({"location": "Paris", "unit": "celsius"})
    assert not _weather_args_ok({})


def test_the_tool_schema_declares_its_required_arguments() -> None:
    assert WEATHER_TOOL["input_schema"]["required"] == ["location", "unit"]


def test_fixtures_live_where_the_default_cli_argument_points() -> None:
    assert Path(__file__).resolve().parents[1] / "fixtures" / "stub" == STUB_FIXTURES
