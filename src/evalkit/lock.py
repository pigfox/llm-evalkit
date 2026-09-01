"""Pre-registration and the lock.

``predict`` writes ``predictions.json`` before anything is run. ``run``,
``grade`` and ``report`` all refuse to proceed unless that file is present and
its recorded hash still matches.

The hash covers two things, and covering both is the point. It covers the
canonicalised predictions block, so the predictions themselves cannot be edited
after seeing a result. And it covers the *source text* of ``graders.py`` and
``cases.py``, so moving the goalposts -- loosening a tolerance, rewording a
prompt, swapping a grader -- invalidates the lock just as loudly as editing a
prediction would. Those are the same act, and a lock that caught only the first
would be a lock you could walk around.

There is no ``--force``. Re-registering means deleting ``predictions.json``
deliberately, which leaves a visible hole in the record rather than a quiet
overwrite.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalkit.cases import Case

LOCK_MESSAGE = "predictions not locked"
DEFAULT_PREDICTIONS_PATH = Path("predictions.json")

#: The source files whose text is bound into the lock.
SOURCE_FILES: tuple[Path, ...] = (
    Path(__file__).with_name("graders.py"),
    Path(__file__).with_name("cases.py"),
)


class LockError(RuntimeError):
    """Raised when predictions are missing, malformed, or no longer match."""


def source_text() -> str:
    """Concatenate the locked source files, each under a stable header."""
    parts: list[str] = []
    for path in SOURCE_FILES:
        parts.append(f"### {path.name}\n{path.read_text(encoding='utf-8')}")
    return "\n".join(parts)


def build_predictions(suite: Sequence[Case]) -> dict[str, Any]:
    """Assemble the predictions block for ``suite``."""
    cases = [
        {
            "id": case.id,
            "predicted_pass": case.predicted_pass,
            "predicted_notes": case.predicted_notes,
        }
        for case in suite
    ]
    predicted_rate = sum(1 for c in cases if c["predicted_pass"]) / len(cases)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": len(cases),
        "predicted_pass_rate": round(predicted_rate, 6),
        "cases": cases,
    }


def compute_lock(payload: dict[str, Any]) -> str:
    """sha256 over the canonicalised payload plus the locked source text."""
    block = {k: v for k, v in payload.items() if k != "lock"}
    digest = hashlib.sha256()
    digest.update(json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(source_text().encode("utf-8"))
    return digest.hexdigest()


def write_predictions(path: Path, suite: Sequence[Case]) -> dict[str, Any]:
    """Pre-register ``suite``. Refuses to overwrite an existing registration."""
    if path.exists():
        raise LockError(
            f"predictions already registered at {path}. Pre-registration is write-once: "
            f"delete the file deliberately if you really mean to re-register, and expect "
            f"the deletion to be visible in your history."
        )
    payload = build_predictions(suite)
    payload["lock"] = compute_lock(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


def load_predictions(path: Path) -> dict[str, Any]:
    """Load and verify a registration, raising :class:`LockError` on any doubt."""
    if not path.exists():
        raise LockError(f"{LOCK_MESSAGE}: no predictions file at {path}. Run `predict` first.")
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LockError(f"{LOCK_MESSAGE}: {path} is not valid JSON ({exc})") from exc
    recorded = payload.get("lock")
    if not isinstance(recorded, str):
        raise LockError(f"{LOCK_MESSAGE}: {path} carries no lock field")
    actual = compute_lock(payload)
    if actual != recorded:
        raise LockError(
            f"{LOCK_MESSAGE}: hash mismatch. Recorded {recorded[:16]}..., computed "
            f"{actual[:16]}.... Either the predictions or the text of graders.py / "
            f"cases.py changed after registration. There is no --force; re-register "
            f"deliberately or restore the files."
        )
    return payload
