"""Deterministic graders.

Every grader in this module is a pure function of the model's output. None of
them calls a model. There is no LLM-judge path here, and adding one behind a
flag would defeat the point of the exercise: a judge model has its own error
rate, its own non-determinism and its own prompt sensitivity, so grading with
one converts a measurement into a comparison between two unmeasured things.

Each grader returns a :class:`Grade` whose ``reason`` is populated on a pass as
well as on a failure, because a report in which only failures explain
themselves is a report you cannot audit for false passes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from evalkit.provider import RawResponse
from evalkit.schemas import Parsed, ParseFailure, ParseResult


@dataclass(frozen=True)
class Grade:
    """The verdict on one model output."""

    passed: bool
    reason: str
    observed: Any
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "observed": self.observed,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class GradeInput:
    """Everything a grader is allowed to look at."""

    parsed: ParseResult
    response: RawResponse


Grader = Callable[[GradeInput], Grade]


def _unparsed(field: str, expected: Any) -> Grade:
    return Grade(
        passed=False,
        reason=f"output did not parse, so field {field!r} could not be read",
        observed=None,
        expected=expected,
    )


def exact_match(field: str, expected: Any) -> Grader:
    """Pass when ``field`` on the parsed value equals ``expected``."""

    def grade(item: GradeInput) -> Grade:
        if isinstance(item.parsed, ParseFailure):
            return _unparsed(field, expected)
        observed = getattr(item.parsed.value, field)
        if observed == expected:
            return Grade(True, f"{field} == {expected!r}", observed, expected)
        return Grade(False, f"{field} was {observed!r}, expected {expected!r}", observed, expected)

    return grade


def numeric_within(field: str, expected: float, tol: float) -> Grader:
    """Pass when ``field`` is within ``tol`` of ``expected``."""

    def grade(item: GradeInput) -> Grade:
        if isinstance(item.parsed, ParseFailure):
            return _unparsed(field, expected)
        observed = float(getattr(item.parsed.value, field))
        delta = abs(observed - expected)
        if delta <= tol:
            return Grade(
                True,
                f"{field} was {observed!r}, within {tol} of {expected!r} (delta {delta:.6g})",
                observed,
                expected,
            )
        return Grade(
            False,
            f"{field} was {observed!r}, outside {tol} of {expected!r} (delta {delta:.6g})",
            observed,
            expected,
        )

    return grade


def field_set_equal(field: str, expected: Iterable[Any]) -> Grader:
    """Pass when ``field`` holds exactly ``expected`` as a set, order ignored."""
    wanted = set(expected)

    def grade(item: GradeInput) -> Grade:
        if isinstance(item.parsed, ParseFailure):
            return _unparsed(field, sorted(map(str, wanted)))
        observed_set = set(getattr(item.parsed.value, field))
        observed = sorted(map(str, observed_set))
        expected_sorted = sorted(map(str, wanted))
        if observed_set == wanted:
            return Grade(True, f"{field} set == {expected_sorted}", observed, expected_sorted)
        missing = sorted(map(str, wanted - observed_set))
        extra = sorted(map(str, observed_set - wanted))
        return Grade(
            False,
            f"{field} set mismatch; missing {missing}, unexpected {extra}",
            observed,
            expected_sorted,
        )

    return grade


def schema_valid(*, expect_valid: bool = True) -> Grader:
    """Pass when the output's parse status matches ``expect_valid``.

    ``expect_valid=False`` is how a case declares that the *correct* outcome is
    a parse failure -- a prompt the model should refuse to answer in JSON, say.
    That keeps ParseFailure a first-class graded result rather than a hole in
    the suite.
    """
    expected = "parsed" if expect_valid else "parse_failure"

    def grade(item: GradeInput) -> Grade:
        valid = isinstance(item.parsed, Parsed)
        observed: dict[str, Any] = {"status": "parsed" if valid else "parse_failure"}
        if isinstance(item.parsed, ParseFailure):
            observed["errors"] = item.parsed.errors
        if valid == expect_valid:
            reason = (
                "output parsed into the case schema"
                if valid
                else "output failed to parse, which is what this case expects"
            )
            return Grade(True, reason, observed, expected)
        reason = (
            "output parsed into the case schema, but this case expects a parse failure"
            if valid
            else "output failed to parse"
        )
        return Grade(False, reason, observed, expected)

    return grade


def tool_call_matches(
    name: str,
    args_predicate: Callable[[dict[str, Any]], bool],
    *,
    predicate_description: str = "the case's argument predicate",
) -> Grader:
    """Pass when the model called ``name`` with satisfying arguments.

    This grader reads ``response.tool_calls`` and nothing else. It never looks
    at prose, so a model that calls the tool correctly and then narrates
    something wrong still passes, and a model that describes the right call in
    prose without making it still fails.
    """

    def grade(item: GradeInput) -> Grade:
        called = [c.name for c in item.response.tool_calls]
        matching = [c for c in item.response.tool_calls if c.name == name]
        expected = {"name": name, "arguments": predicate_description}
        if not matching:
            return Grade(
                False,
                f"no tool call named {name!r}; tools called: {called}",
                {"tool_calls": called},
                expected,
            )
        call = matching[0]
        observed = {"name": call.name, "arguments": call.arguments}
        if args_predicate(call.arguments):
            return Grade(
                True,
                f"tool {name!r} was called with arguments satisfying {predicate_description}",
                observed,
                expected,
            )
        return Grade(
            False,
            f"tool {name!r} was called, but arguments {call.arguments!r} did not satisfy "
            f"{predicate_description}",
            observed,
            expected,
        )

    return grade
