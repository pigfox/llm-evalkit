"""Command line: predict, run, grade, report.

Exit codes are meaningful and are relied on by CI:

* ``0`` -- clean. Note that misses do *not* make this non-zero. A miss is a
  correct, informative outcome of the method; failing the build on one would
  create pressure to predict conservatively, which is the exact bias
  pre-registration exists to remove.
* ``1`` -- lock or usage error. Missing or broken pre-registration, a bad
  argument, no run file to grade.
* ``2`` -- provider failure. The harness worked; the model or its transport did
  not.

``run`` verifies the lock before it calls anything, so you cannot look at
results and then decide what you predicted.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evalkit.cases import SUITE
from evalkit.lock import DEFAULT_PREDICTIONS_PATH, LockError, load_predictions, write_predictions
from evalkit.provider import AnthropicProvider, Provider, StubProvider
from evalkit.report import grade_run, render_markdown, write_markdown, write_results
from evalkit.runner import DEFAULT_K, DEFAULT_RUNS_DIR, latest_run, load_run, run_suite

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_PROVIDER = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="Pre-registered, deterministically graded LLM evaluation.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="path to predictions.json (default: %(default)s)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory holding run files and results (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("predict", help="pre-register predictions. Write-once.")

    run_parser = sub.add_parser("run", help="run the suite k times per case")
    run_parser.add_argument("--provider", choices=("stub", "anthropic"), default="stub")
    run_parser.add_argument("--fixtures", type=Path, default=Path("fixtures/stub"))
    run_parser.add_argument("--model", default=None, help="model id for the anthropic provider")
    run_parser.add_argument("-k", type=int, default=DEFAULT_K)

    sub.add_parser("grade", help="grade the latest run against the lock")
    sub.add_parser("report", help="grade and render markdown + results.json")
    return parser


def _make_provider(args: argparse.Namespace) -> Provider:
    if args.provider == "anthropic":
        model = args.model or "claude-sonnet-5"
        return AnthropicProvider(model_id=model)
    return StubProvider(args.fixtures)


def cmd_predict(args: argparse.Namespace) -> int:
    payload = write_predictions(args.predictions, SUITE)
    print(f"Pre-registered {payload['case_count']} cases to {args.predictions}")
    print(f"  predicted pass rate: {payload['predicted_pass_rate']:.4f}")
    print(f"  lock: {payload['lock']}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    load_predictions(args.predictions)
    provider = _make_provider(args)
    path, payload = run_suite(provider, SUITE, k=args.k, runs_dir=args.runs_dir)
    failed = [
        c["case_id"] for c in payload["cases"] for o in c["outputs"] if o["error"] is not None
    ]
    print(f"Ran {len(payload['cases'])} cases k={payload['k']} against {payload['model_id']}")
    print(f"  run file: {path}")
    if failed:
        print(f"  provider failures on: {sorted(set(failed))}", file=sys.stderr)
        return EXIT_PROVIDER
    return EXIT_OK


def _grade_latest(args: argparse.Namespace) -> dict[str, object]:
    predictions = load_predictions(args.predictions)
    run = load_run(latest_run(args.runs_dir))
    return grade_run(run, SUITE, predictions)


def cmd_grade(args: argparse.Namespace) -> int:
    results = _grade_latest(args)
    results_path = args.runs_dir / "results.json"
    write_results(results, results_path)
    print(f"Graded {results['case_count']} cases from {results['model_id']}")
    print(f"  predicted pass rate: {results['predicted_pass_rate']:.4f}")
    print(f"  observed pass rate:  {results['observed_pass_rate']:.4f}")
    print(f"  delta:               {results['delta']:+.4f}")
    print(f"  misses:              {results['miss_count']}")
    print(f"  results: {results_path}")
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    results = _grade_latest(args)
    results_path = args.runs_dir / "results.json"
    markdown_path = args.runs_dir / "report.md"
    write_results(results, results_path)
    write_markdown(results, markdown_path)
    print(render_markdown(results))
    print(f"[wrote {markdown_path} and {results_path}]", file=sys.stderr)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "predict": cmd_predict,
        "run": cmd_run,
        "grade": cmd_grade,
        "report": cmd_report,
    }
    try:
        return handlers[args.command](args)
    except LockError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PROVIDER if args.command == "run" else EXIT_USAGE
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PROVIDER


if __name__ == "__main__":
    raise SystemExit(main())
