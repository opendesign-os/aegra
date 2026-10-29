"""Contract fields the LangGraph SDK sends on run creation, and the Run response shape."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aegra_api.models.runs import Run, RunCreate


def _payload(**overrides: Any) -> dict[str, Any]:
    return {"assistant_id": "agent", "input": {"messages": []}, **overrides}


class TestDurability:
    """`durability` and its deprecated alias must both reach the runtime."""

    @pytest.mark.parametrize("value", ["sync", "async", "exit"])
    def test_accepts_every_spec_value(self, value: str) -> None:
        assert RunCreate.model_validate(_payload(durability=value)).resolved_durability == value

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            RunCreate.model_validate(_payload(durability="never"))

    def test_checkpoint_during_true_maps_to_async(self) -> None:
        assert RunCreate.model_validate(_payload(checkpoint_during=True)).resolved_durability == "async"

    def test_checkpoint_during_false_maps_to_exit(self) -> None:
        assert RunCreate.model_validate(_payload(checkpoint_during=False)).resolved_durability == "exit"

    def test_omitted_leaves_runtime_default(self) -> None:
        assert RunCreate.model_validate(_payload()).resolved_durability is None

    def test_durability_wins_over_the_deprecated_alias(self) -> None:
        request = RunCreate.model_validate(_payload(durability="sync", checkpoint_during=False))

        assert request.resolved_durability == "sync"


class TestWebhook:
    """The webhook URL is the SSRF entry point, so it is constrained at the boundary."""

    def test_accepts_https(self) -> None:
        request = RunCreate.model_validate(_payload(webhook="https://hooks.example/cb"))

        assert request.webhook == "https://hooks.example/cb"

    @pytest.mark.parametrize("url", ["ftp://host/x", "file:///etc/passwd", "https://", "not-a-url"])
    def test_rejects_non_http_or_hostless(self, url: str) -> None:
        with pytest.raises(ValidationError):
            RunCreate.model_validate(_payload(webhook=url))


class TestSchedulingAndStreamFlags:
    def test_after_seconds_accepts_zero(self) -> None:
        assert RunCreate.model_validate(_payload(after_seconds=0)).after_seconds == 0

    def test_after_seconds_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            RunCreate.model_validate(_payload(after_seconds=-1))

    def test_stream_resumable_defaults_false(self) -> None:
        assert RunCreate.model_validate(_payload()).stream_resumable is False

    def test_stream_resumable_round_trips(self) -> None:
        assert RunCreate.model_validate(_payload(stream_resumable=True)).stream_resumable is True

    @pytest.mark.parametrize("value", ["create", "reject"])
    def test_if_not_exists_accepts_spec_values(self, value: str) -> None:
        assert RunCreate.model_validate(_payload(if_not_exists=value)).if_not_exists == value

    def test_if_not_exists_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            RunCreate.model_validate(_payload(if_not_exists="maybe"))

    def test_feedback_keys_are_accepted(self) -> None:
        request = RunCreate.model_validate(_payload(feedback_keys=["score"]))

        assert request.feedback_keys == ["score"]


class TestSpecKeysAreDeclared:
    """A field the SDK always sends must exist, or it is silently dropped."""

    SDK_ALWAYS_SENT = {"stream_mode", "stream_subgraphs", "stream_resumable", "assistant_id"}
    SDK_OPTIONAL = {
        "input",
        "command",
        "config",
        "context",
        "metadata",
        "interrupt_before",
        "interrupt_after",
        "webhook",
        "checkpoint",
        "checkpoint_id",
        "checkpoint_during",
        "multitask_strategy",
        "if_not_exists",
        "on_completion",
        "after_seconds",
        "durability",
        "feedback_keys",
    }

    def test_every_sdk_key_is_declared(self) -> None:
        missing = (self.SDK_ALWAYS_SENT | self.SDK_OPTIONAL) - set(RunCreate.model_fields)

        assert missing == set(), f"RunCreate would silently drop: {sorted(missing)}"


class _Row:
    """Stand-in for a Run ORM row."""

    def __init__(self, execution_params: dict[str, Any] | None) -> None:
        self.run_id = "run-1"
        self.thread_id = "thread-1"
        self.assistant_id = "agent"
        self.status = "success"
        self.input = {"messages": []}
        self.output = {"ok": True}
        self.error_message = None
        self.config = {}
        self.context = {}
        self.user_id = "test-user"
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.execution_params = execution_params


class TestRunResponseShape:
    """`kwargs`, `metadata` and `multitask_strategy` are projected from execution_params."""

    PARAMS = {
        "execution": {
            "input_data": {"messages": []},
            "config": {"configurable": {"a": 1}},
            "context": {},
            "stream_mode": ["values"],
            "durability": "exit",
            "stream_resumable": True,
        },
        "behavior": {
            "multitask_strategy": "interrupt",
            "interrupt_before": ["node"],
            "interrupt_after": None,
            "webhook": "https://hooks.example/cb",
            "after_seconds": 5,
            "subgraphs": True,
        },
        "run_metadata": {"tenant": "acme"},
    }

    def test_metadata_comes_from_run_metadata(self) -> None:
        assert Run.model_validate(_Row(self.PARAMS)).metadata == {"tenant": "acme"}

    def test_multitask_strategy_comes_from_behavior(self) -> None:
        assert Run.model_validate(_Row(self.PARAMS)).multitask_strategy == "interrupt"

    def test_kwargs_carries_execution_parameters(self) -> None:
        kwargs = Run.model_validate(_Row(self.PARAMS)).kwargs

        assert kwargs["durability"] == "exit"
        assert kwargs["stream_resumable"] is True
        assert kwargs["webhook"] == "https://hooks.example/cb"
        assert kwargs["after_seconds"] == 5
        assert kwargs["interrupt_before"] == ["node"]

    @pytest.mark.parametrize("params", [None, {}])
    def test_legacy_row_without_execution_params_stays_empty(self, params: dict[str, Any] | None) -> None:
        """A row predating execution_params must not surface SQLAlchemy's MetaData as metadata."""
        run = Run.model_validate(_Row(params))

        assert run.metadata == {}
        assert run.kwargs == {}
        assert run.multitask_strategy is None
