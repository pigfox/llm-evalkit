"""k runs per case, all kept, nothing voted on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit.cases import SUITE, by_id
from evalkit.provider import AttemptRecord, ProviderError, Request, StubProvider
from evalkit.runner import (
    CaseRun,
    agreement_key,
    latest_run,
    load_run,
    parse_output,
    run_case,
    run_suite,
)
from evalkit.schemas import Parsed, ParseFailure


class ExplodingProvider:
    """A provider whose every attempt fails, with the chain attached."""

    model_id = "exploding"

    def complete(self, request: Request):
        raise ProviderError(
            "every attempt failed after 2 recorded attempt(s)",
            [
                AttemptRecord(1, "error", 3, "OverloadedError", "529"),
                AttemptRecord(2, "timeout", 4, "TimeoutError", "read timed out"),
            ],
        )


def stub(workspace: Path) -> StubProvider:
    return StubProvider(workspace / "fixtures" / "stub")


def test_parse_output_uses_the_case_schema() -> None:
    result = parse_output(
        by_id("sentiment_clear_positive"), '{"label":"positive","confidence":0.9}'
    )
    assert isinstance(result, Parsed)


def test_a_case_without_a_schema_reports_that_rather_than_a_fake_json_error() -> None:
    result = parse_output(by_id("weather_tool_call"), "I will look that up.")
    assert isinstance(result, ParseFailure)
    assert result.errors[0]["type"] == "no_schema"
    assert "graded on tool calls" in result.errors[0]["msg"]


def test_agreement_key_is_stable_across_prose_differences() -> None:
    case = by_id("sentiment_clear_positive")
    a = agreement_key(case, 'Sure: {"label":"positive","confidence":0.9}', [])
    b = agreement_key(case, '```json\n{"label": "positive", "confidence": 0.9}\n```', [])
    assert a == b


def test_agreement_key_separates_different_parsed_values() -> None:
    case = by_id("sentiment_clear_positive")
    a = agreement_key(case, '{"label":"positive","confidence":0.9}', [])
    b = agreement_key(case, '{"label":"negative","confidence":0.9}', [])
    assert a != b


def test_agreement_key_falls_back_to_raw_text_when_nothing_parses() -> None:
    case = by_id("refusal_is_not_json")
    key = agreement_key(case, "  a paragraph of prose  ", [])
    assert json.loads(key)["unparsed"] == "a paragraph of prose"


def test_agreement_key_distinguishes_tool_calls() -> None:
    case = by_id("weather_tool_call")
    a = agreement_key(case, "x", [{"name": "get_weather", "arguments": {"unit": "celsius"}}])
    b = agreement_key(case, "x", [{"name": "get_weather", "arguments": {"unit": "fahrenheit"}}])
    assert a != b


def test_case_run_serialises() -> None:
    run = CaseRun(case_id="c", outputs=[{"index": 1}], agreement_rate=1.0)
    assert run.to_dict() == {"case_id": "c", "agreement_rate": 1.0, "outputs": [{"index": 1}]}


def test_run_case_keeps_all_k_outputs_and_never_collapses_them(workspace: Path) -> None:
    result = run_case(stub(workspace), by_id("sentiment_unstable"), k=3)
    assert len(result.outputs) == 3
    labels = [json.loads(o["text"])["label"] for o in result.outputs]
    assert labels == ["positive", "negative", "positive"]
    assert result.agreement_rate < 1.0


def test_run_case_reports_full_agreement_for_a_stable_case(workspace: Path) -> None:
    result = run_case(stub(workspace), by_id("sentiment_clear_positive"), k=3)
    assert result.agreement_rate == 1.0


def test_run_case_records_the_retry_chain(workspace: Path) -> None:
    result = run_case(stub(workspace), by_id("sentiment_retry_then_ok"), k=1)
    output = result.outputs[0]
    assert output["attempt_count"] == 2
    assert [a["outcome"] for a in output["attempts"]] == ["error", "ok"]
    assert output["error"] is None


def test_a_provider_failure_is_recorded_as_an_output_not_raised(workspace: Path) -> None:
    result = run_case(ExplodingProvider(), by_id("sentiment_clear_positive"), k=2)
    assert len(result.outputs) == 2
    for output in result.outputs:
        assert output["error"]["type"] == "ProviderError"
        assert output["attempt_count"] == 2
        assert [a["outcome"] for a in output["attempts"]] == ["error", "timeout"]


def test_run_suite_writes_a_run_file_with_env_and_model_metadata(workspace: Path) -> None:
    path, payload = run_suite(stub(workspace), SUITE, k=2, runs_dir=workspace / "runs")
    assert path.exists()
    assert path.name.startswith("run-")
    assert payload["k"] == 2
    assert payload["model_id"] == "stub-fixture-replay"
    assert payload["provider_class"] == "StubProvider"
    assert payload["env"]["python"]
    assert payload["env"]["platform"]
    assert len(payload["cases"]) == len(SUITE)
    assert json.loads(path.read_text()) == payload


def test_run_suite_rejects_a_k_below_one(workspace: Path) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        run_suite(stub(workspace), SUITE, k=0, runs_dir=workspace / "runs")


def test_latest_run_picks_the_newest_stamp(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    for stamp in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
        (runs / f"run-{stamp}.json").write_text("{}")
    assert latest_run(runs).name == "run-20260301T000000Z.json"


def test_latest_run_says_what_to_do_when_there_are_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run `run` first"):
        latest_run(tmp_path)


def test_load_run_reads_a_run_file(workspace: Path) -> None:
    path, payload = run_suite(stub(workspace), SUITE, k=1, runs_dir=workspace / "runs")
    assert load_run(path) == payload
