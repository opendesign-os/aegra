"""Agent-to-Agent protocol endpoint: JSON-RPC 2.0 over ``POST /a2a/{assistant_id}``.

Envelopes, task models and error types all come from ``a2a.compat.v0_3.types``.
Dispatch is local rather than ``JsonRpcDispatcher`` because the SDK's v0.3
compat adapter catches every handler exception and reports it as
``InternalError``, losing the spec error codes (-32002, -32004, -32005) that
A2A clients branch on.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from a2a.compat.v0_3 import types as a2a
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.sse import get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import User
from aegra_api.services import a2a_service
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

router = APIRouter(tags=["A2A"], dependencies=auth_dependency)

_REQUEST_ADAPTER: TypeAdapter[a2a.A2ARequest] = TypeAdapter(a2a.A2ARequest)
_POLL_INTERVAL_SECS = 0.5
_SUPPORTED_METHODS = ["message/send", "message/stream", "tasks/cancel", "tasks/get"]

# The body is parsed by hand so malformed JSON answers with a JSON-RPC error
# envelope instead of FastAPI's 422; the schema is declared here for the docs.
_REQUEST_SCHEMA: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["jsonrpc", "id", "method"],
                    "properties": {
                        "jsonrpc": {"type": "string", "enum": ["2.0"]},
                        "id": {"type": "string"},
                        "method": {"type": "string", "enum": _SUPPORTED_METHODS},
                        "params": {
                            "type": "object",
                            "description": "Method parameters; shape depends on the method.",
                        },
                    },
                }
            }
        },
    }
}


def _wire(model: BaseModel) -> dict[str, Any]:
    """Serialize an envelope, keeping ``id`` even when null as JSON-RPC requires."""
    body = model.model_dump(by_alias=True, exclude_none=True, mode="json")
    body.setdefault("id", None)
    return body


def _error(request_id: str | int | None, error: Any) -> JSONResponse:
    return JSONResponse(_wire(a2a.JSONRPCErrorResponse(id=request_id, error=error)))


@router.post("/a2a/{assistant_id}", openapi_extra=_REQUEST_SCHEMA)
async def a2a_rpc(
    assistant_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> Any:
    """Dispatch one A2A JSON-RPC envelope against an assistant."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(None, a2a.JSONParseError())

    if not isinstance(payload, dict):
        return _error(None, a2a.InvalidRequestError())

    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0":
        return _error(request_id, a2a.InvalidRequestError())

    try:
        envelope = _REQUEST_ADAPTER.validate_python(payload).root
    except ValidationError:
        method = payload.get("method")
        if not isinstance(method, str):
            return _error(request_id, a2a.InvalidRequestError())
        if method not in _SUPPORTED_METHODS:
            return _error(request_id, a2a.MethodNotFoundError())
        # A known method that failed the union means its params are wrong.
        return _error(request_id, a2a.InvalidParamsError())

    handlers = {
        a2a.SendMessageRequest: _message_send,
        a2a.SendStreamingMessageRequest: _message_stream,
        a2a.GetTaskRequest: _tasks_get,
        a2a.CancelTaskRequest: _tasks_cancel,
    }
    handler = handlers.get(type(envelope))
    if handler is None:
        return _error(request_id, a2a.UnsupportedOperationError())

    try:
        return await handler(assistant_id, envelope, user, payload)
    except Exception:
        logger.exception("A2A dispatch failed", method=envelope.method, assistant_id=assistant_id)
        return _error(request_id, a2a.InternalError())


async def _guard_message(
    envelope: a2a.SendMessageRequest | a2a.SendStreamingMessageRequest, user: User
) -> JSONResponse | None:
    message = envelope.params.message
    unsupported = a2a_service.unsupported_part_kinds(message)
    if unsupported:
        return _error(
            envelope.id,
            a2a.ContentTypeNotSupportedError(message=f"Unsupported part kinds: {', '.join(unsupported)}"),
        )
    if message.context_id and await a2a_service.thread_is_foreign(message.context_id, user):
        return _error(envelope.id, a2a.InvalidParamsError(message="Unknown contextId"))
    return None


async def _message_send(
    assistant_id: str, envelope: a2a.SendMessageRequest, user: User, _payload: dict[str, Any]
) -> JSONResponse:
    rejected = await _guard_message(envelope, user)
    if rejected is not None:
        return rejected

    task_id, context_id = await a2a_service.start_task(assistant_id, envelope.params.message, user)
    task = await a2a_service.wait_for_task(task_id, context_id, user)
    return JSONResponse(_wire(a2a.SendMessageSuccessResponse(id=envelope.id, result=task)))


async def _message_stream(
    assistant_id: str, envelope: a2a.SendStreamingMessageRequest, user: User, _payload: dict[str, Any]
) -> Any:
    rejected = await _guard_message(envelope, user)
    if rejected is not None:
        return rejected

    task_id, context_id = await a2a_service.start_task(assistant_id, envelope.params.message, user)

    async def envelopes() -> AsyncGenerator[str, None]:
        deadline = asyncio.get_running_loop().time() + settings.worker.BG_JOB_TIMEOUT_SECS
        last_state: a2a.TaskState | None = None
        while True:
            task = await a2a_service.get_task(task_id, context_id, user)
            if task is None:
                yield _sse(_wire(a2a.JSONRPCErrorResponse(id=envelope.id, error=a2a.TaskNotFoundError())))
                return
            if task.status.state != last_state:
                last_state = task.status.state
                yield _sse(_wire(a2a.SendStreamingMessageSuccessResponse(id=envelope.id, result=task)))
            if task.status.state in a2a_service.TERMINAL_STATES:
                return
            if asyncio.get_running_loop().time() >= deadline:
                yield _sse(_wire(a2a.JSONRPCErrorResponse(id=envelope.id, error=a2a.InternalError())))
                return
            await asyncio.sleep(_POLL_INTERVAL_SECS)

    return make_sse_response(sse_to_bytes(envelopes()), headers=get_sse_headers())


async def _tasks_get(
    _assistant_id: str, envelope: a2a.GetTaskRequest, user: User, payload: dict[str, Any]
) -> JSONResponse:
    # contextId is an upstream addition to TaskQueryParams: Aegra needs the
    # thread to scope the lookup, and the base A2A spec does not carry it.
    context_id = (payload.get("params") or {}).get("contextId")
    if not isinstance(context_id, str) or not context_id:
        return _error(envelope.id, a2a.InvalidParamsError(message="contextId is required"))

    task = await a2a_service.get_task(envelope.params.id, context_id, user)
    if task is None:
        return _error(envelope.id, a2a.TaskNotFoundError())
    return JSONResponse(_wire(a2a.GetTaskSuccessResponse(id=envelope.id, result=task)))


async def _tasks_cancel(
    _assistant_id: str, envelope: a2a.CancelTaskRequest, _user: User, _payload: dict[str, Any]
) -> JSONResponse:
    return _error(envelope.id, a2a.TaskNotCancelableError())


def _sse(envelope: dict[str, Any]) -> str:
    return f"data: {json.dumps(envelope)}\n\n"
