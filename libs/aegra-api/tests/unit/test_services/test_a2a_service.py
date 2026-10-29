"""A2A wire contract and the mapping between A2A tasks and Aegra runs.

The models come from ``a2a.compat.v0_3.types`` — the SDK's JSON-RPC shapes.
``a2a.types`` is the protobuf/gRPC representation and is a different wire form.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from a2a.compat.v0_3 import types as a2a
from pydantic import TypeAdapter, ValidationError

from aegra_api.services.a2a_service import (
    TERMINAL_STATES,
    message_to_input,
    run_to_task,
    unsupported_part_kinds,
)

REQUEST_ADAPTER: TypeAdapter[a2a.A2ARequest] = TypeAdapter(a2a.A2ARequest)


def _message(**overrides: Any) -> a2a.Message:
    payload = {
        "role": "user",
        "parts": [{"kind": "text", "text": "hello"}],
        "messageId": "m-1",
        **overrides,
    }
    return a2a.Message.model_validate(payload)


class _Run:
    def __init__(self, status: str, output: Any = None, error: str | None = None) -> None:
        self.run_id = "run-1"
        self.thread_id = "ctx-1"
        self.status = status
        self.output = output
        self.error_message = error
        self.updated_at = datetime.now(UTC)


class TestWireShape:
    """Field names and enum values must be the JSON spec form, not the protobuf one."""

    def test_message_serialises_camel_case(self) -> None:
        body = _message(contextId="ctx-9").model_dump(by_alias=True, exclude_none=True, mode="json")

        assert body["contextId"] == "ctx-9"
        assert body["messageId"] == "m-1"
        assert body["role"] == "user"
        assert body["parts"][0] == {"kind": "text", "text": "hello"}

    def test_task_serialises_camel_case(self) -> None:
        body = run_to_task(_Run("success")).model_dump(by_alias=True, exclude_none=True, mode="json")

        assert body["contextId"] == "ctx-1"
        assert body["kind"] == "task"
        assert "context_id" not in body

    def test_task_states_are_the_hyphenated_spec_values(self) -> None:
        assert a2a.TaskState.input_required.value == "input-required"
        assert a2a.TaskState.completed.value == "completed"


class TestRequestParsing:
    @pytest.mark.parametrize(
        ("method", "params", "expected"),
        [
            (
                "message/send",
                {"message": {"role": "user", "parts": [{"kind": "text", "text": "x"}], "messageId": "m"}},
                a2a.SendMessageRequest,
            ),
            (
                "message/stream",
                {"message": {"role": "user", "parts": [{"kind": "text", "text": "x"}], "messageId": "m"}},
                a2a.SendStreamingMessageRequest,
            ),
            ("tasks/get", {"id": "t-1"}, a2a.GetTaskRequest),
            ("tasks/cancel", {"id": "t-1"}, a2a.CancelTaskRequest),
        ],
    )
    def test_each_method_parses_to_its_typed_request(self, method: str, params: dict[str, Any], expected: type) -> None:
        envelope = REQUEST_ADAPTER.validate_python(
            {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
        ).root

        assert isinstance(envelope, expected)

    def test_wrong_jsonrpc_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            REQUEST_ADAPTER.validate_python({"jsonrpc": "1.0", "id": "1", "method": "tasks/get", "params": {"id": "t"}})

    def test_file_parts_parse_but_are_reported_as_unsupported(self) -> None:
        """The base spec allows file parts; this endpoint cannot carry them, so it says so."""
        message = a2a.Message.model_validate(
            {"role": "user", "parts": [{"kind": "file", "file": {"uri": "x"}}], "messageId": "m"}
        )

        assert unsupported_part_kinds(message) == ["file"]

    def test_text_and_data_parts_are_supported(self) -> None:
        message = a2a.Message.model_validate(
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "x"}, {"kind": "data", "data": {"a": 1}}],
                "messageId": "m",
            }
        )

        assert unsupported_part_kinds(message) == []

    def test_message_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            a2a.Message.model_validate({"role": "user", "parts": [{"kind": "text", "text": "x"}]})


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
            ("pending", a2a.TaskState.submitted),
            ("running", a2a.TaskState.working),
            ("success", a2a.TaskState.completed),
            ("error", a2a.TaskState.failed),
            ("timeout", a2a.TaskState.failed),
            ("interrupted", a2a.TaskState.input_required),
        ],
    )
    def test_status_mapping(self, run_status: str, task_state: a2a.TaskState) -> None:
        assert run_to_task(_Run(run_status)).status.state == task_state

    def test_unknown_status_falls_back_to_working(self) -> None:
        assert run_to_task(_Run("something-new")).status.state == a2a.TaskState.working

    def test_output_becomes_a_data_artifact(self) -> None:
        task = run_to_task(_Run("success", output={"answer": 42}))

        assert task.artifacts is not None
        assert task.artifacts[0].parts[0].root.data == {"answer": 42}

    def test_no_output_means_no_artifacts(self) -> None:
        assert run_to_task(_Run("running")).artifacts is None

    def test_error_is_surfaced_as_an_agent_message(self) -> None:
        task = run_to_task(_Run("error", error="boom"))

        assert task.status.message is not None
        assert task.status.message.parts[0].root.text == "boom"
        assert task.status.message.role == a2a.Role.agent

    def test_terminal_states_cover_the_finished_run_statuses(self) -> None:
        for status in ("success", "error", "timeout"):
            assert run_to_task(_Run(status)).status.state in TERMINAL_STATES

    def test_interrupted_is_not_terminal(self) -> None:
        """An interrupted run is waiting on input, so the stream must keep going."""
        assert run_to_task(_Run("interrupted")).status.state not in TERMINAL_STATES


class TestErrorCodes:
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (a2a.TaskNotFoundError(), -32001),
            (a2a.TaskNotCancelableError(), -32002),
            (a2a.MethodNotFoundError(), -32601),
            (a2a.InvalidParamsError(), -32602),
            (a2a.InvalidRequestError(), -32600),
            (a2a.JSONParseError(), -32700),
            (a2a.InternalError(), -32603),
        ],
    )
    def test_sdk_errors_carry_the_spec_codes(self, error: Any, code: int) -> None:
        assert error.code == code
