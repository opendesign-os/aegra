"""Agent-to-Agent (A2A) JSON-RPC 2.0 wire models.

The A2A JSON transport uses lowerCamelCase field names and hyphenated task
states, which is a different representation from the protobuf/gRPC one.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

A2AMethod = Literal["message/send", "message/stream", "tasks/get", "tasks/cancel"]

TaskState = Literal[
    "submitted",
    "working",
    "completed",
    "failed",
    "canceled",
    "input-required",
    "rejected",
    "auth-required",
]

# JSON-RPC 2.0 reserved codes, plus the A2A range for protocol-specific failures.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
UNSUPPORTED_OPERATION = -32004


class TextPart(BaseModel):
    kind: Literal["text"]
    text: str


class DataPart(BaseModel):
    kind: Literal["data"]
    data: dict[str, Any]


Part = TextPart | DataPart


class A2AMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "agent"]
    parts: list[Part]
    message_id: str = Field(alias="messageId")
    context_id: str | None = Field(None, alias="contextId")
    task_id: str | None = Field(None, alias="taskId")

    @model_validator(mode="after")
    def require_a_supported_part(self) -> Self:
        if not self.parts:
            raise ValueError("message.parts must contain at least one text or data part")
        return self


class MessageParams(BaseModel):
    message: A2AMessage


class TaskParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    context_id: str = Field(alias="contextId")
    history_length: int | None = Field(None, alias="historyLength", ge=0, le=10)


class TaskStatus(BaseModel):
    state: TaskState
    timestamp: str | None = None
    message: dict[str, Any] | None = None


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    context_id: str = Field(alias="contextId")
    status: TaskStatus
    kind: Literal["task"] = "task"
    artifacts: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str | int
    method: str
    params: dict[str, Any] | None = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    result: Any | None = None
    error: JSONRPCError | None = None


def success(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def failure(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    """``id`` stays in the envelope even when null: JSON-RPC requires it."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
