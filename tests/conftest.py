"""Shared fixtures. Every test in this suite runs against StubProvider."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_FIXTURES = REPO_ROOT / "fixtures" / "stub"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated cwd holding a copy of the stub fixtures."""
    shutil.copytree(STUB_FIXTURES, tmp_path / "fixtures" / "stub")
    monkeypatch.chdir(tmp_path)
    return tmp_path
