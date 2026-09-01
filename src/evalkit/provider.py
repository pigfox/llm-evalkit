"""Providers and the retry machinery around them.

Two things matter here and both are about visibility rather than throughput.

First, every attempt is recorded. A provider that silently retried three times
before answering looks, in a naive harness, exactly like one that answered
immediately -- and the difference is the whole story when you are trying to say
something honest about a model's reliability. ``RawResponse.attempts`` carries
one record per attempt, successes and failures alike, and the run file keeps
them.

Second, the Anthropic SDK has its own retry loop, enabled by default. It is
switched off here (``max_retries=0``) so that the attempt records are complete;
otherwise the SDK would absorb retries below the level this module can see.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anthropic import APITimeoutError


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt at a request, whatever its outcome.

    ``outcome`` is one of ``ok``, ``error``, ``timeout`` or ``deadline``.
    """

    index: int
    outcome: str
    duration_ms: int
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class Request:
    """A single completion request for one case."""

    case_id: str
    prompt: str
    system: str | None = None
    tools: tuple[dict[str, Any], ...] = ()
    max_tokens: int = 1024


@dataclass(frozen=True)
class RawResponse:
    """What a provider returned, including how many tries it took."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str | None
    latency_ms: int
    attempt_count: int
    attempts: tuple[AttemptRecord, ...]


class Provider(Protocol):
    """The one thing a provider must do."""

    model_id: str

    def complete(self, request: Request) -> RawResponse: ...


class ProviderError(RuntimeError):
    """Raised when every permitted attempt failed. Carries the whole chain."""

    def __init__(self, message: str, attempts: Sequence[AttemptRecord]) -> None:
        super().__init__(message)
        self.attempts: tuple[AttemptRecord, ...] = tuple(attempts)


class StubScriptError(RuntimeError):
    """A failure scripted by a stub fixture, or a fixture that ran out."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded attempts, exponential backoff, per-attempt and overall limits."""

    max_attempts: int = 3
    initial_backoff_s: float = 0.25
    backoff_multiplier: float = 2.0
    per_attempt_timeout_s: float = 30.0
    overall_deadline_s: float = 90.0

    def backoff_for(self, attempt_index: int) -> float:
        """Seconds to wait after attempt ``attempt_index`` (1-based)."""
        return self.initial_backoff_s * (self.backoff_multiplier ** (attempt_index - 1))


#: A single attempt. Receives the timeout it must honour, returns
#: ``(text, tool_calls, stop_reason)``, and signals a timeout by raising
#: ``TimeoutError``.
AttemptCall = Callable[[float], tuple[str, tuple[ToolCall, ...], str | None]]


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def execute_with_retry(
    call: AttemptCall,
    policy: RetryPolicy,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RawResponse:
    """Run ``call`` under ``policy``, recording every attempt.

    The per-attempt timeout is passed *to* the call, which is responsible for
    honouring it -- the HTTP client is the only layer that can actually abandon
    an in-flight request, so this function does not pretend to interrupt one.
    What it does guarantee is that no attempt is started once the overall
    deadline has passed, and that a call which reports a timeout is recorded as
    a timeout rather than as a generic error.

    ``clock`` and ``sleep`` are injectable so tests exercise backoff and
    deadline behaviour without spending wall-clock time.
    """
    started = clock()
    attempts: list[AttemptRecord] = []
    for index in range(1, policy.max_attempts + 1):
        elapsed = clock() - started
        if elapsed >= policy.overall_deadline_s:
            attempts.append(
                AttemptRecord(
                    index=index,
                    outcome="deadline",
                    duration_ms=0,
                    error_type="DeadlineExceeded",
                    error_message=(
                        f"overall deadline of {policy.overall_deadline_s}s was already "
                        f"exhausted before attempt {index} began"
                    ),
                )
            )
            break
        timeout = min(policy.per_attempt_timeout_s, policy.overall_deadline_s - elapsed)
        attempt_started = clock()
        try:
            text, tool_calls, stop_reason = call(timeout)
        except TimeoutError as exc:
            attempts.append(
                AttemptRecord(
                    index=index,
                    outcome="timeout",
                    duration_ms=_ms(clock() - attempt_started),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        except Exception as exc:
            attempts.append(
                AttemptRecord(
                    index=index,
                    outcome="error",
                    duration_ms=_ms(clock() - attempt_started),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        else:
            attempts.append(
                AttemptRecord(
                    index=index,
                    outcome="ok",
                    duration_ms=_ms(clock() - attempt_started),
                )
            )
            return RawResponse(
                text=text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                latency_ms=_ms(clock() - started),
                attempt_count=index,
                attempts=tuple(attempts),
            )
        if index < policy.max_attempts:
            sleep(policy.backoff_for(index))
    raise ProviderError(f"every attempt failed after {len(attempts)} recorded attempt(s)", attempts)


def _tool_calls_from_specs(specs: Sequence[dict[str, Any]]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(name=str(spec.get("name", "")), arguments=dict(spec.get("arguments", {})))
        for spec in specs
    )


class StubProvider:
    """Replays recorded fixtures from disk. Fully deterministic.

    One JSON file per case, ``<fixture_dir>/<case_id>.json``::

        {"case_id": "x", "calls": [
            {"text": "...", "tool_calls": [], "stop_reason": "end_turn"},
            {"attempts": [{"error": {"type": "OverloadedError", "message": "529"}},
                          {"text": "...", "stop_reason": "end_turn"}]}
        ]}

    A call is either a single successful response or an explicit list of
    per-attempt outcomes, which is how the default suite exercises the retry
    path without a network. The nth call to :meth:`complete` for a case replays
    ``calls[n % len(calls)]``, so k repetitions are reproducible and a case can
    be scripted to disagree with itself.
    """

    def __init__(
        self,
        fixture_dir: Path | str,
        *,
        model_id: str = "stub-fixture-replay",
        policy: RetryPolicy | None = None,
    ) -> None:
        self.fixture_dir = Path(fixture_dir)
        self.model_id = model_id
        self.policy = policy or RetryPolicy(initial_backoff_s=0.0)
        self._cursor: dict[str, int] = {}

    def _calls_for(self, case_id: str) -> list[dict[str, Any]]:
        path = self.fixture_dir / f"{case_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no stub fixture for case {case_id!r} at {path}")
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        calls: list[dict[str, Any]] = payload["calls"]
        if not calls:
            raise StubScriptError(f"fixture for case {case_id!r} declares no calls")
        return calls

    @staticmethod
    def _attempt_specs(call_spec: dict[str, Any]) -> list[dict[str, Any]]:
        attempts = call_spec.get("attempts")
        if isinstance(attempts, list):
            return [dict(a) for a in attempts]
        return [call_spec]

    def complete(self, request: Request) -> RawResponse:
        calls = self._calls_for(request.case_id)
        index = self._cursor.get(request.case_id, 0)
        self._cursor[request.case_id] = index + 1
        remaining = self._attempt_specs(calls[index % len(calls)])

        def call(_timeout: float) -> tuple[str, tuple[ToolCall, ...], str | None]:
            if not remaining:
                raise StubScriptError(
                    f"fixture for case {request.case_id!r} ran out of scripted attempts"
                )
            spec = remaining.pop(0)
            error = spec.get("error")
            if error is not None:
                kind = str(error.get("type", "StubScriptError"))
                message = str(error.get("message", ""))
                if kind == "TimeoutError":
                    raise TimeoutError(message)
                raise StubScriptError(f"{kind}: {message}")
            return (
                str(spec.get("text", "")),
                _tool_calls_from_specs(spec.get("tool_calls", [])),
                spec.get("stop_reason", "end_turn"),
            )

        return execute_with_retry(call, self.policy, sleep=lambda _s: None)


class AnthropicProvider:
    """Talks to the real API. Reads its key from the environment only.

    Written against anthropic 1.2.0, which is what ``requirements-lock.txt``
    pins. Two details of that version are load-bearing: ``messages.create`` no
    longer accepts ``temperature``, so this provider does not pass one and
    makes no determinism claim on that basis; and the client retries twice by
    default, which is disabled here so that ``RawResponse.attempts`` is the
    complete record.
    """

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model_id: str = "claude-sonnet-5",
        *,
        policy: RetryPolicy | None = None,
        client: Any | None = None,
    ) -> None:
        api_key = os.environ.get(self.ENV_VAR, "")
        if not api_key:
            raise RuntimeError(
                f"{self.ENV_VAR} is not set. AnthropicProvider reads it from os.environ "
                f"only and never opens a file; export {self.ENV_VAR} and retry."
            )
        self.model_id = model_id
        self.policy = policy or RetryPolicy()
        self._client = self._build_client(api_key) if client is None else client

    @staticmethod
    def _build_client(api_key: str) -> Any:
        from anthropic import Anthropic

        return Anthropic(api_key=api_key, max_retries=0)

    @staticmethod
    def _extract(message: Any) -> tuple[str, tuple[ToolCall, ...], str | None]:
        """Split a Message into prose and tool calls.

        Content blocks are read by their ``type`` discriminator rather than by
        position, because a response mixes text and ``tool_use`` blocks freely
        and future block types must be ignored rather than misread.
        """
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in getattr(message, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(str(getattr(block, "text", "")))
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", None) or {}),
                    )
                )
        stop_reason = getattr(message, "stop_reason", None)
        return "\n".join(texts), tuple(calls), stop_reason

    def complete(self, request: Request) -> RawResponse:
        def call(timeout: float) -> tuple[str, tuple[ToolCall, ...], str | None]:
            kwargs: dict[str, Any] = {
                "model": self.model_id,
                "max_tokens": request.max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
                "timeout": timeout,
            }
            if request.system is not None:
                kwargs["system"] = request.system
            if request.tools:
                kwargs["tools"] = list(request.tools)
            try:
                message = self._client.messages.create(**kwargs)
            except APITimeoutError as exc:
                raise TimeoutError(str(exc)) from exc
            return self._extract(message)

        return execute_with_retry(call, self.policy)
