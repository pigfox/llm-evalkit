"""The report names every miss, in both directions, before anything else."""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.cases import SUITE, by_id
from evalkit.lock import build_predictions
from evalkit.provider import StubProvider
from evalkit.report import (
    grade_output,
    grade_run,
    render_markdown,
    write_markdown,
    write_results,
)
from evalkit.runner import run_suite


def graded(workspace: Path, k: int = 3) -> dict:
    _path, run = run_suite(
        StubProvider(workspace / "fixtures" / "stub"), SUITE, k=k, runs_dir=workspace / "runs"
    )
    return grade_run(run, SUITE, build_predictions(SUITE))


def test_grade_output_rebuilds_the_response_including_tool_calls(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    from evalkit.provider import Request

    response = provider.complete(Request(case_id="weather_tool_call", prompt="x"))
    output = {
        "text": response.text,
        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
        "stop_reason": response.stop_reason,
        "latency_ms": response.latency_ms,
        "attempt_count": response.attempt_count,
        "attempts": [a.to_dict() for a in response.attempts],
    }
    grade = grade_output(by_id("weather_tool_call"), output)
    assert grade.passed


def test_a_case_passes_only_when_every_one_of_the_k_outputs_passes(workspace: Path) -> None:
    results = graded(workspace)
    unstable = next(c for c in results["cases"] if c["id"] == "sentiment_unstable")
    passed = [g["passed"] for g in unstable["grades"]]
    assert passed == [True, False, True]
    assert unstable["observed_pass"] is False, "two out of three is not a pass"


def test_grade_run_reports_rates_and_the_delta(workspace: Path) -> None:
    results = graded(workspace)
    assert results["case_count"] == len(SUITE)
    assert results["predicted_pass_rate"] == 0.875
    assert results["observed_pass_rate"] == 0.75
    assert results["delta"] == -0.125
    assert results["miss_count"] == 2


def test_the_report_emits_a_miss_when_observed_differs_from_predicted(workspace: Path) -> None:
    results = graded(workspace)
    misses = {c["id"] for c in results["cases"] if c["is_miss"]}
    assert misses == {"invoice_line_items", "sentiment_unstable"}

    markdown = render_markdown(results)
    misses_section = markdown.split("## Summary")[0]
    assert "## Misses (2)" in misses_section
    for case_id in misses:
        assert f"`{case_id}`" in misses_section
    assert misses_section.index("## Misses") < markdown.index("## Summary")


def test_a_predicted_failure_that_fails_is_not_a_miss(workspace: Path) -> None:
    results = graded(workspace)
    row = next(c for c in results["cases"] if c["id"] == "measurement_unit_literal")
    assert row["predicted_pass"] is False
    assert row["observed_pass"] is False
    assert row["is_miss"] is False


def test_a_miss_is_reported_in_the_other_direction_too(workspace: Path) -> None:
    results = graded(workspace)
    row = next(c for c in results["cases"] if c["id"] == "measurement_unit_literal")
    row["observed_pass"] = True
    row["is_miss"] = True
    markdown = render_markdown(results)
    assert "predicted FAIL, observed pass" in markdown


def test_the_report_lists_cases_that_disagree_with_themselves(workspace: Path) -> None:
    markdown = render_markdown(graded(workspace))
    assert "## Cases with agreement below 1.0 (1)" in markdown
    assert "`sentiment_unstable` — agreement 0.33" in markdown


def test_a_clean_report_says_so_in_both_sections(workspace: Path) -> None:
    results = graded(workspace)
    for row in results["cases"]:
        row["is_miss"] = False
        row["agreement_rate"] = 1.0
    markdown = render_markdown(results)
    assert "Every case matched its pre-registered prediction." in markdown
    assert "Every case returned the same structured output on all 3 runs." in markdown


def test_the_summary_table_carries_every_case(workspace: Path) -> None:
    markdown = render_markdown(graded(workspace))
    table = markdown.split("## Summary")[1].split("## Pass rate")[0]
    for case in SUITE:
        assert f"`{case.id}`" in table


def test_the_pass_rate_block_states_predicted_observed_and_delta(workspace: Path) -> None:
    markdown = render_markdown(graded(workspace))
    assert "- Predicted: 87.5%" in markdown
    assert "- Observed: 75.0%" in markdown
    assert "- Delta: -12.5 percentage points" in markdown
    assert "- Misses: 2 of 16" in markdown


def test_results_and_markdown_are_written_to_disk(workspace: Path) -> None:
    results = graded(workspace)
    results_path = workspace / "out" / "results.json"
    markdown_path = workspace / "out" / "report.md"
    write_results(results, results_path)
    write_markdown(results, markdown_path)
    assert json.loads(results_path.read_text()) == results
    assert markdown_path.read_text().startswith("# llm-evalkit results")


def test_the_retry_case_reports_more_attempts_than_outputs(workspace: Path) -> None:
    results = graded(workspace)
    row = next(c for c in results["cases"] if c["id"] == "sentiment_retry_then_ok")
    assert row["attempt_total"] == 6
    assert "| 6 |" in render_markdown(results)
