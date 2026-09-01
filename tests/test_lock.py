"""Pre-registration is write-once, and the lock covers the graders too."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit import lock as lock_module
from evalkit.cases import SUITE
from evalkit.lock import (
    LOCK_MESSAGE,
    LockError,
    build_predictions,
    compute_lock,
    load_predictions,
    source_text,
    write_predictions,
)


@pytest.fixture
def fake_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Stand-in graders.py / cases.py so a test can mutate one safely."""
    graders = tmp_path / "graders.py"
    cases = tmp_path / "cases.py"
    graders.write_text("def exact_match(): return True\n")
    cases.write_text("SUITE = ()\n")
    monkeypatch.setattr(lock_module, "SOURCE_FILES", (graders, cases))
    return graders, cases


def test_source_text_covers_both_locked_files() -> None:
    text = source_text()
    assert "### graders.py" in text
    assert "### cases.py" in text
    assert "def exact_match(" in text
    assert "SUITE" in text


def test_build_predictions_records_every_case_and_the_rate() -> None:
    payload = build_predictions(SUITE)
    assert payload["case_count"] == len(SUITE)
    assert len(payload["cases"]) == len(SUITE)
    expected = sum(1 for c in SUITE if c.predicted_pass) / len(SUITE)
    assert payload["predicted_pass_rate"] == pytest.approx(expected)
    assert {"id", "predicted_pass", "predicted_notes"} == set(payload["cases"][0])


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    written = write_predictions(path, SUITE)
    loaded = load_predictions(path)
    assert loaded == written
    assert len(loaded["lock"]) == 64


def test_pre_registration_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    write_predictions(path, SUITE)
    with pytest.raises(LockError, match="write-once"):
        write_predictions(path, SUITE)


def test_grade_refuses_when_predictions_are_absent(tmp_path: Path) -> None:
    with pytest.raises(LockError) as excinfo:
        load_predictions(tmp_path / "predictions.json")
    assert str(excinfo.value).startswith(LOCK_MESSAGE)
    assert "Run `predict` first" in str(excinfo.value)


def test_grade_refuses_when_the_file_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text("{not json")
    with pytest.raises(LockError) as excinfo:
        load_predictions(path)
    assert str(excinfo.value).startswith(LOCK_MESSAGE)


def test_grade_refuses_when_the_lock_field_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(build_predictions(SUITE)))
    with pytest.raises(LockError) as excinfo:
        load_predictions(path)
    assert str(excinfo.value).startswith(LOCK_MESSAGE)
    assert "carries no lock field" in str(excinfo.value)


def test_grade_refuses_when_a_prediction_was_edited_after_locking(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    write_predictions(path, SUITE)
    payload = json.loads(path.read_text())
    payload["cases"][0]["predicted_pass"] = not payload["cases"][0]["predicted_pass"]
    path.write_text(json.dumps(payload))
    with pytest.raises(LockError) as excinfo:
        load_predictions(path)
    assert str(excinfo.value).startswith(LOCK_MESSAGE)
    assert "hash mismatch" in str(excinfo.value)


def test_grade_refuses_after_graders_py_is_mutated_post_lock(
    tmp_path: Path, fake_sources: tuple[Path, Path]
) -> None:
    graders, _cases = fake_sources
    path = tmp_path / "predictions.json"
    write_predictions(path, SUITE)
    load_predictions(path)  # clean before the edit

    graders.write_text("def exact_match(): return True  # loosened after the fact\n")

    with pytest.raises(LockError) as excinfo:
        load_predictions(path)
    assert str(excinfo.value).startswith(LOCK_MESSAGE)
    assert "graders.py" in str(excinfo.value)
    assert "no --force" in str(excinfo.value)


def test_grade_refuses_after_cases_py_is_mutated_post_lock(
    tmp_path: Path, fake_sources: tuple[Path, Path]
) -> None:
    _graders, cases = fake_sources
    path = tmp_path / "predictions.json"
    write_predictions(path, SUITE)
    cases.write_text("SUITE = ()  # a prompt quietly reworded\n")
    with pytest.raises(LockError):
        load_predictions(path)


def test_the_lock_ignores_its_own_field_when_hashing(tmp_path: Path) -> None:
    payload = build_predictions(SUITE)
    without = compute_lock(payload)
    payload["lock"] = "0" * 64
    assert compute_lock(payload) == without
