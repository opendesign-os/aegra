"""A2A wire models and the mapping between A2A tasks and Aegra runs."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aegra_api.models.a2a import (
    INVALID_PARAMS,
    A2AMessage,
    JSONRPCRequest,
    MessageParams,
    Task,
    TaskParams,
    failure,
    success,
)
from aegra_api.services.a2a_service import message_to_input, run_to_task


def _message(**overrides: Any) -> A2AMessage:
    payload = {
        "role": "user",
        "parts": [{"kind": "text", "text": "hello"}],
        "messageId": "m-1",
        **overrides,
    }
    return A2AMessage.model_validate(payload)


class _Run:
    def __init__(self, status: str, output: Any = None, error: str | None = None) -> None:
        self.run_id = "run-1"
        self.thread_id = "ctx-1"
        self.status = status
        self.output = output
        self.error_message = error
        self.updated_at = datetime.now(UTC)


class TestMessageModel:
    def test_camel_case_fields_are_the_wire_form(self) -> None:
        message = _message(contextId="ctx-9", taskId="t-9")

        assert message.context_id == "ctx-9"
        assert message.task_id == "t-9"

    def test_message_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            A2AMessage.model_validate({"role": "user", "parts": [{"kind": "text", "text": "x"}]})

    def test_empty_parts_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            A2AMessage.model_validate({"role": "user", "parts": [], "messageId": "m-1"})

    def test_file_parts_are_not_accepted(self) -> None:
        """Only text and data parts are carried; a file part must not pass silently."""
        with pytest.raises(ValidationError):
            A2AMessage.model_validate({"role": "user", "parts": [{"kind": "file", "uri": "x"}], "messageId": "m-1"})

    @pytest.mark.parametrize("role", ["user", "agent"])
    def test_spec_roles_are_accepted(self, role: str) -> None:
        assert _message(role=role).role == role

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _message(role="system")


class TestMessageToInput:
    def test_text_becomes_a_message_entry(self) -> None:
        assert message_to_input(_message()) == {"messages": [{"role": "user", "content": "hello"}]}

    def test_agent_role_maps_to_assistant(self) -> None:
        payload = message_to_input(_message(role="agent"))

        assert payload["messages"][0]["role"] == "assistant"

    def test_data_parts_merge_into_the_input(self) -> None:
        message = _message(parts=[{"kind": "data", "data": {"topic": "weather"}}])

        assert message_to_input(message) == {"topic": "weather"}

    def test_text_and_data_combine(self) -> None:
        message = _message(parts=[{"kind": "text", "text": "hi"}, {"kind": "data", "data": {"topic": "weather"}}])
        payload = message_to_input(message)

        assert payload["topic"] == "weather"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]


class TestRunToTask:
    @pytest.mark.parametrize(
        ("run_status", "task_state"),
        [
            ("pending", "submitted"),
            ("running", "working"),
            ("success", "completed"),
            ("error", "failed"),
            ("timeout", "failed"),
            ("interrupted", "input-required"),
        ],
    )
    def test_status_mapping(self, run_status: str, task_state: str) -> None:
        assert run_to_task(_Run(run_status)).status.state == task_state

    def test_unknown_status_falls_back_to_working(self) -> None:
        assert run_to_task(_Run("something-new")).status.state == "working"

    def test_output_becomes_a_data_artifact(self) -> None:
        task = run_to_task(_Run("success", output={"answer": 42}))

        assert task.artifacts is not None
        assert task.artifacts[0]["parts"][0] == {"kind": "data", "data": {"answer": 42}}

    def test_no_output_means_no_artifacts(self) -> None:
        assert run_to_task(_Run("running")).artifacts is None

    def test_error_is_surfaced_on_the_status(self) -> None:
        task = run_to_task(_Run("error", error="boom"))

        assert task.status.message == {"error": "boom"}

    def test_serialises_with_the_spec_field_names(self) -> None:
        body = run_to_task(_Run("success")).model_dump(exclude_none=True)

        assert body["contextId"] == "ctx-1"
        assert body["kind"] == "task"
        assert "context_id" not in body


class TestParamModels:
    def test_task_params_use_camel_case(self) -> None:
        params = TaskParams.model_validate({"id": "t-1", "contextId": "ctx-1", "historyLength": 3})

        assert params.context_id == "ctx-1"
        assert params.history_length == 3

    def test_history_length_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            TaskParams.model_validate({"id": "t-1", "contextId": "c", "historyLength": 99})

    def test_message_params_require_a_message(self) -> None:
        with pytest.raises(ValidationError):
            MessageParams.model_validate({})


class TestEnvelopes:
    def test_request_requires_jsonrpc_two(self) -> None:
        with pytest.raises(ValidationError):
            JSONRPCRequest.model_validate({"jsonrpc": "1.0", "id": "1", "method": "tasks/get"})

    def test_success_envelope_omits_the_error_slot(self) -> None:
        body = success("1", {"ok": True})

        assert body == {"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}

    def test_failure_envelope_omits_the_result_slot(self) -> None:
        body = failure("1", INVALID_PARAMS, "bad")

        assert body == {"jsonrpc": "2.0", "id": "1", "error": {"code": INVALID_PARAMS, "message": "bad"}}

    def test_failure_keeps_a_null_id_when_the_request_was_unparseable(self) -> None:
        assert failure(None, INVALID_PARAMS, "bad")["id"] is None


class TestTaskModel:
    def test_kind_defaults_to_task(self) -> None:
        task = Task.model_validate({"id": "t", "contextId": "c", "status": {"state": "completed"}})

        assert task.kind == "task"
