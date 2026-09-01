"""Runs the suite k times per case and records everything.

The runner does not decide anything. It does not majority-vote across the k
outputs, it does not pick a canonical answer, and it does not average a score.
All k outputs are kept and all k are graded later; a case passes only if every
one of its outputs passes. ``agreement_rate`` reports how often the model
agreed with itself, and it is reported alongside the pass rate rather than
folded into it, because "right once out of three" and "right three times out of
three" are different facts about a model and collapsing them destroys the more
interesting one.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalkit import __version__
from evalkit.cases import Case
from evalkit.provider import Provider, ProviderError, Request
from evalkit.schemas import Parsed, ParseFailure, ParseResult, parse_into

DEFAULT_K = 3
DEFAULT_RUNS_DIR = Path("runs")


def parse_output(case: Case, text: str) -> ParseResult:
    """Parse one output under the case's schema.

    A case with no schema is graded on its tool calls alone. It still gets a
    parse result so the grader interface stays uniform, and that result says so
    rather than pretending the prose was malformed JSON.
    """
    if case.schema is None:
        return ParseFailure(
            raw=text,
            errors=[
                {
                    "type": "no_schema",
                    "loc": [],
                    "msg": "this case declares no schema; it is graded on tool calls",
                }
            ],
        )
    return parse_into(case.schema, text)


def agreement_key(case: Case, text: str, tool_calls: Sequence[dict[str, Any]]) -> str:
    """A canonical form of one output, for counting self-agreement.

    Two outputs agree when they parse to the same object and request the same
    tool calls. Prose differences below that do not count as disagreement --
    the suite grades structure, so it measures agreement on structure.
    """
    parsed = parse_output(case, text)
    if isinstance(parsed, Parsed):
        body: Any = {"parsed": parsed.value.model_dump(mode="json")}
    else:
        body = {"unparsed": text.strip()}
    body["tool_calls"] = [{"name": c["name"], "arguments": c["arguments"]} for c in tool_calls]
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


@dataclass
class CaseRun:
    """Every output recorded for one case."""

    case_id: str
    outputs: list[dict[str, Any]]
    agreement_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agreement_rate": self.agreement_rate,
            "outputs": self.outputs,
        }


def run_case(provider: Provider, case: Case, k: int) -> CaseRun:
    """Call the provider k times for one case, recording each result."""
    outputs: list[dict[str, Any]] = []
    for index in range(1, k + 1):
        request = Request(case_id=case.id, prompt=case.prompt, system=case.system, tools=case.tools)
        try:
            response = provider.complete(request)
        except ProviderError as exc:
            outputs.append(
                {
                    "index": index,
                    "text": "",
                    "tool_calls": [],
                    "stop_reason": None,
                    "latency_ms": 0,
                    "attempt_count": len(exc.attempts),
                    "attempts": [a.to_dict() for a in exc.attempts],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            continue
        outputs.append(
            {
                "index": index,
                "text": response.text,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
                ],
                "stop_reason": response.stop_reason,
                "latency_ms": response.latency_ms,
                "attempt_count": response.attempt_count,
                "attempts": [a.to_dict() for a in response.attempts],
                "error": None,
            }
        )
    keys = Counter(agreement_key(case, o["text"], o["tool_calls"]) for o in outputs)
    modal = keys.most_common(1)[0][1]
    return CaseRun(case_id=case.id, outputs=outputs, agreement_rate=modal / len(outputs))


def run_suite(
    provider: Provider,
    suite: Sequence[Case],
    *,
    k: int = DEFAULT_K,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    now: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the whole suite and write the run file. Returns its path and payload."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    started = now or datetime.now(UTC)
    case_runs = [run_case(provider, case, k) for case in suite]
    finished = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evalkit_version": __version__,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "k": k,
        "model_id": provider.model_id,
        "provider_class": type(provider).__name__,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "cases": [cr.to_dict() for cr in case_runs],
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"run-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, payload


def latest_run(runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """The most recent run file, by name (the stamp sorts lexicographically)."""
    candidates = sorted(runs_dir.glob("run-*.json"))
    if not candidates:
        raise FileNotFoundError(f"no run files in {runs_dir}. Run `run` first.")
    return candidates[-1]


def load_run(path: Path) -> dict[str, Any]:
    """Read a run file."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
