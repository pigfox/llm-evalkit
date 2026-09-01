"""Grades a run against the lock and renders the result.

Two rules are structural rather than stylistic.

A case passes only if *all* k of its outputs pass. Nothing is majority-voted,
so a model that answers correctly twice and wrongly once has not passed; it has
produced a case with an agreement rate of 0.67 and a failure.

Every miss gets its own section, printed before the summary table. A miss is a
case where the observed outcome disagrees with the pre-registered prediction,
in either direction -- predicted-pass-observed-fail and
predicted-fail-observed-pass are both misses, because both mean the prediction
was wrong, and the second is the one a report is otherwise tempted to quietly
bank as a win. There is no flag that hides the section.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evalkit.cases import Case
from evalkit.graders import Grade, GradeInput
from evalkit.provider import AttemptRecord, RawResponse, ToolCall
from evalkit.runner import parse_output


def _response_from_output(output: dict[str, Any]) -> RawResponse:
    return RawResponse(
        text=output["text"],
        tool_calls=tuple(
            ToolCall(name=c["name"], arguments=dict(c["arguments"])) for c in output["tool_calls"]
        ),
        stop_reason=output["stop_reason"],
        latency_ms=output["latency_ms"],
        attempt_count=output["attempt_count"],
        attempts=tuple(AttemptRecord(**a) for a in output["attempts"]),
    )


def grade_output(case: Case, output: dict[str, Any]) -> Grade:
    """Grade one recorded output of one case."""
    parsed = parse_output(case, output["text"])
    response = _response_from_output(output)
    return case.grader(GradeInput(parsed=parsed, response=response))


def grade_run(
    run: dict[str, Any], suite: Sequence[Case], predictions: dict[str, Any]
) -> dict[str, Any]:
    """Grade a whole run against a verified pre-registration."""
    predicted = {c["id"]: c for c in predictions["cases"]}
    by_case = {c["case_id"]: c for c in run["cases"]}
    rows: list[dict[str, Any]] = []
    for case in suite:
        recorded = by_case[case.id]
        grades = [grade_output(case, output) for output in recorded["outputs"]]
        observed_pass = all(g.passed for g in grades)
        predicted_pass = bool(predicted[case.id]["predicted_pass"])
        rows.append(
            {
                "id": case.id,
                "predicted_pass": predicted_pass,
                "predicted_notes": predicted[case.id]["predicted_notes"],
                "observed_pass": observed_pass,
                "is_miss": observed_pass != predicted_pass,
                "agreement_rate": recorded["agreement_rate"],
                "attempt_total": sum(o["attempt_count"] for o in recorded["outputs"]),
                "grades": [g.to_dict() for g in grades],
            }
        )
    observed_rate = sum(1 for r in rows if r["observed_pass"]) / len(rows)
    predicted_rate = float(predictions["predicted_pass_rate"])
    return {
        "schema_version": 1,
        "run_started_at": run["started_at"],
        "model_id": run["model_id"],
        "provider_class": run["provider_class"],
        "k": run["k"],
        "case_count": len(rows),
        "predicted_pass_rate": round(predicted_rate, 6),
        "observed_pass_rate": round(observed_rate, 6),
        "delta": round(observed_rate - predicted_rate, 6),
        "miss_count": sum(1 for r in rows if r["is_miss"]),
        "cases": rows,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _tick(value: bool) -> str:
    return "pass" if value else "FAIL"


def render_markdown(results: dict[str, Any]) -> str:
    """Render the results as markdown, misses first."""
    lines: list[str] = []
    lines.append("# llm-evalkit results")
    lines.append("")
    lines.append(f"- Model: `{results['model_id']}` via `{results['provider_class']}`")
    lines.append(f"- Run started: {results['run_started_at']}")
    lines.append(f"- Cases: {results['case_count']}, each run k={results['k']} times")
    lines.append("- A case passes only if all k outputs pass. Nothing is majority-voted.")
    lines.append("")

    misses = [r for r in results["cases"] if r["is_miss"]]
    lines.append(f"## Misses ({len(misses)})")
    lines.append("")
    if not misses:
        lines.append("Every case matched its pre-registered prediction.")
        lines.append("")
    else:
        lines.append(
            "Cases where the observed outcome disagreed with the pre-registered "
            "prediction, in either direction."
        )
        lines.append("")
        for row in misses:
            direction = (
                "predicted pass, observed FAIL"
                if row["predicted_pass"]
                else "predicted FAIL, observed pass"
            )
            lines.append(f"### `{row['id']}` — {direction}")
            lines.append("")
            lines.append(f"- Registered reasoning: {row['predicted_notes']}")
            lines.append(f"- Agreement rate: {row['agreement_rate']:.2f}")
            for i, grade in enumerate(row["grades"], start=1):
                lines.append(f"- Output {i}: {_tick(grade['passed'])} — {grade['reason']}")
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| case | predicted | observed | miss | agreement | attempts |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in results["cases"]:
        lines.append(
            f"| `{row['id']}` | {_tick(row['predicted_pass'])} | {_tick(row['observed_pass'])} "
            f"| {'yes' if row['is_miss'] else ''} | {row['agreement_rate']:.2f} "
            f"| {row['attempt_total']} |"
        )
    lines.append("")

    lines.append("## Pass rate")
    lines.append("")
    lines.append(f"- Predicted: {_pct(results['predicted_pass_rate'])}")
    lines.append(f"- Observed: {_pct(results['observed_pass_rate'])}")
    lines.append(f"- Delta: {results['delta'] * 100:+.1f} percentage points")
    lines.append(f"- Misses: {results['miss_count']} of {results['case_count']}")
    lines.append("")

    unstable = [r for r in results["cases"] if r["agreement_rate"] < 1.0]
    lines.append(f"## Cases with agreement below 1.0 ({len(unstable)})")
    lines.append("")
    if not unstable:
        lines.append(f"Every case returned the same structured output on all {results['k']} runs.")
    else:
        for row in unstable:
            lines.append(
                f"- `{row['id']}` — agreement {row['agreement_rate']:.2f}; "
                f"the model did not agree with itself across k={results['k']} runs."
            )
    lines.append("")
    return "\n".join(lines)


def write_results(results: dict[str, Any], path: Path) -> None:
    """Write results.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def write_markdown(results: dict[str, Any], path: Path) -> None:
    """Write the markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results), encoding="utf-8")
