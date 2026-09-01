"""End to end through the CLI, on StubProvider, with meaningful exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit.__main__ import EXIT_OK, EXIT_PROVIDER, EXIT_USAGE, _make_provider, build_parser, main
from evalkit.cases import SUITE
from evalkit.provider import AnthropicProvider, StubProvider

STUB = ["--runs-dir", "runs"]


def test_predict_writes_a_locked_registration(
    workspace: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["predict"]) == EXIT_OK
    payload = json.loads((workspace / "predictions.json").read_text())
    assert payload["case_count"] == len(SUITE)
    assert len(payload["lock"]) == 64
    assert "Pre-registered 16 cases" in capsys.readouterr().out


def test_predict_refuses_to_overwrite_a_registration(workspace: Path) -> None:
    assert main(["predict"]) == EXIT_OK
    assert main(["predict"]) == EXIT_USAGE


def test_run_refuses_without_a_registration(workspace: Path, capsys: pytest.CaptureFixture) -> None:
    assert main([*STUB, "run"]) == EXIT_USAGE
    assert "predictions not locked" in capsys.readouterr().err
    assert not (workspace / "runs").exists(), "nothing may run before pre-registration"


def test_grade_refuses_without_a_registration(
    workspace: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main([*STUB, "grade"]) == EXIT_USAGE
    assert "predictions not locked" in capsys.readouterr().err


def test_grade_refuses_after_a_prediction_is_edited(workspace: Path) -> None:
    assert main(["predict"]) == EXIT_OK
    path = workspace / "predictions.json"
    payload = json.loads(path.read_text())
    payload["cases"][0]["predicted_pass"] = False
    path.write_text(json.dumps(payload))
    assert main([*STUB, "grade"]) == EXIT_USAGE


def test_grade_without_a_run_is_a_usage_error(
    workspace: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["predict"]) == EXIT_OK
    assert main([*STUB, "grade"]) == EXIT_USAGE
    assert "Run `run` first" in capsys.readouterr().err


def test_the_full_pipeline_runs_clean(workspace: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["predict"]) == EXIT_OK
    assert main([*STUB, "run", "-k", "3"]) == EXIT_OK
    assert main([*STUB, "grade"]) == EXIT_OK
    assert main([*STUB, "report"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "observed pass rate:  0.7500" in out
    assert "## Misses (2)" in out

    results = json.loads((workspace / "runs" / "results.json").read_text())
    assert results["miss_count"] == 2
    markdown = (workspace / "runs" / "report.md").read_text()
    assert markdown.index("## Misses") < markdown.index("## Summary")


def test_run_reports_a_provider_failure_with_exit_code_two(
    workspace: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["predict"]) == EXIT_OK
    fixture = workspace / "fixtures" / "stub" / "sentiment_clear_positive.json"
    fixture.write_text(
        json.dumps(
            {
                "case_id": "sentiment_clear_positive",
                "calls": [{"attempts": [{"error": {"type": "OverloadedError", "message": "529"}}]}],
            }
        )
    )
    assert main([*STUB, "run", "-k", "1"]) == EXIT_PROVIDER
    assert "provider failures on: ['sentiment_clear_positive']" in capsys.readouterr().err


def test_a_missing_fixture_directory_is_a_provider_failure(
    workspace: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["predict"]) == EXIT_OK
    assert main([*STUB, "run", "--fixtures", "nowhere"]) == EXIT_PROVIDER
    assert "no stub fixture" in capsys.readouterr().err


def test_a_missing_api_key_is_a_provider_failure_not_a_crash(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["predict"]) == EXIT_OK
    assert main([*STUB, "run", "--provider", "anthropic"]) == EXIT_PROVIDER
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_make_provider_defaults_to_the_stub(workspace: Path) -> None:
    args = build_parser().parse_args([*STUB, "run"])
    provider = _make_provider(args)
    assert isinstance(provider, StubProvider)
    assert provider.fixture_dir == Path("fixtures/stub")


def test_make_provider_builds_the_anthropic_provider_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    args = build_parser().parse_args(["run", "--provider", "anthropic", "--model", "claude-opus-5"])
    provider = _make_provider(args)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model_id == "claude-opus-5"


def test_make_provider_has_a_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    args = build_parser().parse_args(["run", "--provider", "anthropic"])
    assert _make_provider(args).model_id == "claude-sonnet-5"


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
