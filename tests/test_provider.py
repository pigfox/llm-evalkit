"""Retries are recorded, not swallowed; timeouts are recorded, not hangs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import APITimeoutError

from evalkit.provider import (
    AnthropicProvider,
    AttemptRecord,
    ProviderError,
    RawResponse,
    Request,
    RetryPolicy,
    StubProvider,
    StubScriptError,
    ToolCall,
    execute_with_retry,
)

FAST = RetryPolicy(max_attempts=3, initial_backoff_s=0.0, overall_deadline_s=90.0)


class StepClock:
    """A monotonic clock that advances a fixed amount per read."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def ok_call(text: str = "hi"):
    def call(_timeout: float):
        return text, (), "end_turn"

    return call


def test_backoff_is_exponential_from_the_first_attempt() -> None:
    policy = RetryPolicy(initial_backoff_s=0.5, backoff_multiplier=2.0)
    assert policy.backoff_for(1) == 0.5
    assert policy.backoff_for(2) == 1.0
    assert policy.backoff_for(3) == 2.0


def test_attempt_record_serialises() -> None:
    record = AttemptRecord(index=1, outcome="ok", duration_ms=12)
    assert record.to_dict() == {
        "index": 1,
        "outcome": "ok",
        "duration_ms": 12,
        "error_type": None,
        "error_message": None,
    }


def test_a_clean_first_attempt_records_exactly_one_attempt() -> None:
    response = execute_with_retry(ok_call("done"), FAST, clock=StepClock(), sleep=lambda _s: None)
    assert response.text == "done"
    assert response.attempt_count == 1
    assert [a.outcome for a in response.attempts] == ["ok"]


def test_the_retry_path_records_every_attempt_including_the_failures() -> None:
    seen: list[float] = []

    def call(_timeout: float):
        if len(seen) < 2:
            seen.append(1.0)
            raise RuntimeError(f"overloaded #{len(seen)}")
        return "recovered", (), "end_turn"

    slept: list[float] = []
    response = execute_with_retry(
        call,
        RetryPolicy(max_attempts=3, initial_backoff_s=0.5),
        clock=StepClock(),
        sleep=slept.append,
    )

    assert response.text == "recovered"
    assert response.attempt_count == 3
    assert [a.outcome for a in response.attempts] == ["error", "error", "ok"]
    assert response.attempts[0].error_type == "RuntimeError"
    assert response.attempts[0].error_message == "overloaded #1"
    assert response.attempts[1].error_message == "overloaded #2"
    assert slept == [0.5, 1.0]


def test_a_timeout_surfaces_as_a_recorded_failure_rather_than_a_hang() -> None:
    def call(_timeout: float):
        raise TimeoutError("read timed out after 30.0s")

    with pytest.raises(ProviderError) as excinfo:
        execute_with_retry(call, FAST, clock=StepClock(), sleep=lambda _s: None)

    attempts = excinfo.value.attempts
    assert len(attempts) == 3
    assert {a.outcome for a in attempts} == {"timeout"}
    assert attempts[0].error_type == "TimeoutError"
    assert "read timed out" in str(attempts[0].error_message)


def test_the_per_attempt_timeout_is_clamped_by_the_remaining_deadline() -> None:
    handed: list[float] = []

    def call(timeout: float):
        handed.append(timeout)
        return "ok", (), "end_turn"

    policy = RetryPolicy(per_attempt_timeout_s=30.0, overall_deadline_s=5.0)
    execute_with_retry(call, policy, clock=StepClock(), sleep=lambda _s: None)
    assert handed == [5.0]


def test_an_exhausted_overall_deadline_stops_before_starting_an_attempt() -> None:
    called = False

    def call(_timeout: float):
        nonlocal called
        called = True
        return "never", (), "end_turn"

    policy = RetryPolicy(max_attempts=3, overall_deadline_s=0.0)
    with pytest.raises(ProviderError) as excinfo:
        execute_with_retry(call, policy, clock=StepClock(), sleep=lambda _s: None)

    assert called is False
    assert [a.outcome for a in excinfo.value.attempts] == ["deadline"]
    assert excinfo.value.attempts[0].error_type == "DeadlineExceeded"


def test_latency_is_measured_across_the_whole_call_not_one_attempt() -> None:
    def call(_timeout: float):
        return "ok", (), "end_turn"

    response = execute_with_retry(call, FAST, clock=StepClock(step=1.0), sleep=lambda _s: None)
    assert response.latency_ms > 0


# --- StubProvider -----------------------------------------------------------


def test_stub_provider_replays_a_fixture_deterministically(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    request = Request(case_id="sentiment_clear_positive", prompt="x")
    first = provider.complete(request)
    second = provider.complete(request)
    assert first.text == second.text
    assert provider.model_id == "stub-fixture-replay"


def test_stub_provider_cycles_through_scripted_calls(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    request = Request(case_id="sentiment_unstable", prompt="x")
    labels = [json.loads(provider.complete(request).text)["label"] for _ in range(4)]
    assert labels == ["positive", "negative", "positive", "positive"]


def test_stub_provider_scripts_a_retry_and_records_both_attempts(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    response = provider.complete(Request(case_id="sentiment_retry_then_ok", prompt="x"))
    assert response.attempt_count == 2
    assert [a.outcome for a in response.attempts] == ["error", "ok"]
    assert "529" in str(response.attempts[0].error_message)


def test_stub_provider_returns_tool_calls(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    response = provider.complete(Request(case_id="weather_tool_call", prompt="x"))
    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"location": "Reykjavik", "unit": "celsius"}


def test_stub_provider_reports_a_missing_fixture_by_path(tmp_path: Path) -> None:
    provider = StubProvider(tmp_path)
    with pytest.raises(FileNotFoundError, match="no stub fixture for case 'nope'"):
        provider.complete(Request(case_id="nope", prompt="x"))


def test_stub_provider_rejects_a_fixture_with_no_calls(tmp_path: Path) -> None:
    (tmp_path / "empty.json").write_text(json.dumps({"case_id": "empty", "calls": []}))
    provider = StubProvider(tmp_path)
    with pytest.raises(StubScriptError, match="declares no calls"):
        provider.complete(Request(case_id="empty", prompt="x"))


def test_a_scripted_timeout_is_recorded_as_a_timeout(tmp_path: Path) -> None:
    (tmp_path / "slow.json").write_text(
        json.dumps(
            {
                "case_id": "slow",
                "calls": [
                    {
                        "attempts": [
                            {"error": {"type": "TimeoutError", "message": "took too long"}},
                            {"text": "{}", "stop_reason": "end_turn"},
                        ]
                    }
                ],
            }
        )
    )
    provider = StubProvider(tmp_path)
    response = provider.complete(Request(case_id="slow", prompt="x"))
    assert [a.outcome for a in response.attempts] == ["timeout", "ok"]


def test_a_fixture_that_runs_out_of_attempts_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "short.json").write_text(
        json.dumps(
            {
                "case_id": "short",
                "calls": [{"attempts": [{"error": {"type": "OverloadedError", "message": "529"}}]}],
            }
        )
    )
    provider = StubProvider(tmp_path, policy=RetryPolicy(max_attempts=2, initial_backoff_s=0.0))
    with pytest.raises(ProviderError) as excinfo:
        provider.complete(Request(case_id="short", prompt="x"))
    assert "ran out of scripted attempts" in str(excinfo.value.attempts[1].error_message)


# --- AnthropicProvider ------------------------------------------------------


class FakeBlock:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeMessage:
    def __init__(self, content: list[Any], stop_reason: str | None) -> None:
        self.content = content
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, results: list[Any]) -> None:
        self.messages = FakeMessages(results)


def test_a_missing_api_key_raises_at_construction_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_an_empty_api_key_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="never opens a file"):
        AnthropicProvider()


def test_the_real_client_is_built_from_the_environment_without_touching_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    provider = AnthropicProvider(model_id="claude-sonnet-5")
    assert provider.model_id == "claude-sonnet-5"
    # Disabling the SDK's own retries is what makes the attempt record complete.
    assert provider._client.max_retries == 0


def test_anthropic_provider_splits_text_from_tool_use_and_ignores_unknown_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    message = FakeMessage(
        content=[
            FakeBlock(type="text", text="Looking that up."),
            FakeBlock(type="thinking", thinking="ignored entirely"),
            FakeBlock(type="tool_use", name="get_weather", input={"location": "Reykjavik"}),
            FakeBlock(type="text", text="Done."),
        ],
        stop_reason="tool_use",
    )
    client = FakeClient([message])
    provider = AnthropicProvider(model_id="claude-sonnet-5", client=client)
    response = provider.complete(
        Request(case_id="c", prompt="p", system="s", tools=({"name": "get_weather"},))
    )

    assert response.text == "Looking that up.\nDone."
    assert response.tool_calls == (
        ToolCall(name="get_weather", arguments={"location": "Reykjavik"}),
    )
    assert response.stop_reason == "tool_use"
    sent = client.messages.calls[0]
    assert sent["system"] == "s"
    assert sent["tools"] == [{"name": "get_weather"}]
    assert sent["timeout"] == 30.0
    # anthropic 1.2.0 removed `temperature` from messages.create.
    assert "temperature" not in sent


def test_anthropic_provider_omits_system_and_tools_when_a_case_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    client = FakeClient([FakeMessage(content=[], stop_reason="end_turn")])
    provider = AnthropicProvider(client=client)
    response = provider.complete(Request(case_id="c", prompt="p"))
    assert response.text == ""
    assert "system" not in client.messages.calls[0]
    assert "tools" not in client.messages.calls[0]


def test_an_sdk_timeout_is_translated_into_a_recorded_timeout_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(
        [
            APITimeoutError(request=request),
            FakeMessage(
                content=[FakeBlock(type="text", text="second time lucky")], stop_reason="end_turn"
            ),
        ]
    )
    provider = AnthropicProvider(
        client=client, policy=RetryPolicy(max_attempts=2, initial_backoff_s=0.0)
    )
    response = provider.complete(Request(case_id="c", prompt="p"))
    assert response.text == "second time lucky"
    assert [a.outcome for a in response.attempts] == ["timeout", "ok"]


def test_raw_response_is_a_plain_value(workspace: Path) -> None:
    provider = StubProvider(workspace / "fixtures" / "stub")
    response = provider.complete(Request(case_id="sentiment_clear_positive", prompt="x"))
    assert isinstance(response, RawResponse)
    assert response.tool_calls == ()
