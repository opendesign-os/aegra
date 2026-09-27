"""Agent-to-Agent protocol endpoint: JSON-RPC 2.0 over ``POST /a2a/{assistant_id}``."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.sse import get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import User
from aegra_api.models.a2a import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    JSONRPCRequest,
    MessageParams,
    TaskParams,
    failure,
    success,
)
from aegra_api.services import a2a_service
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

router = APIRouter(tags=["A2A"], dependencies=auth_dependency)

_TERMINAL_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
_POLL_INTERVAL_SECS = 0.5


@router.post("/a2a/{assistant_id}")
async def a2a_rpc(
    assistant_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> Any:
    """Dispatch one A2A JSON-RPC envelope against an assistant."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(failure(None, INVALID_REQUEST, "Request body is not valid JSON."))

    try:
        envelope = JSONRPCRequest.model_validate(payload)
    except ValidationError as exc:
        request_id = payload.get("id") if isinstance(payload, dict) else None
        return JSONResponse(failure(request_id, INVALID_REQUEST, str(exc.errors()[0].get("msg", "invalid request"))))

    handlers = {
        "message/send": _message_send,
        "message/stream": _message_stream,
        "tasks/get": _tasks_get,
        "tasks/cancel": _tasks_cancel,
    }
    handler = handlers.get(envelope.method)
    if handler is None:
        return JSONResponse(failure(envelope.id, METHOD_NOT_FOUND, f"Unknown method {envelope.method!r}."))

    try:
        return await handler(assistant_id, envelope, user)
    except Exception:
        logger.exception("A2A dispatch failed", method=envelope.method, assistant_id=assistant_id)
        return JSONResponse(failure(envelope.id, INTERNAL_ERROR, "Request failed unexpectedly."))


async def _message_send(assistant_id: str, envelope: JSONRPCRequest, user: User) -> JSONResponse:
    params, error = _message_params(envelope)
    if params is None:
        return error

    if params.message.context_id and await a2a_service.thread_is_foreign(params.message.context_id, user):
        return JSONResponse(failure(envelope.id, INVALID_PARAMS, "Unknown contextId."))

    task_id, context_id = await a2a_service.start_task(assistant_id, params.message, user)
    task = await a2a_service.wait_for_task(task_id, context_id, user)
    return JSONResponse(success(envelope.id, task.model_dump(exclude_none=True)))


async def _message_stream(assistant_id: str, envelope: JSONRPCRequest, user: User) -> Any:
    params, error = _message_params(envelope)
    if params is None:
        return error

    if params.message.context_id and await a2a_service.thread_is_foreign(params.message.context_id, user):
        return JSONResponse(failure(envelope.id, INVALID_PARAMS, "Unknown contextId."))

    task_id, context_id = await a2a_service.start_task(assistant_id, params.message, user)

    async def envelopes() -> AsyncGenerator[str, None]:
        deadline = asyncio.get_running_loop().time() + settings.worker.BG_JOB_TIMEOUT_SECS
        last_state: str | None = None
        while True:
            task = await a2a_service.get_task(task_id, context_id, user)
            if task is None:
                yield _sse(failure(envelope.id, TASK_NOT_FOUND, "Task not found."))
                return
            if task.status.state != last_state:
                last_state = task.status.state
                yield _sse(success(envelope.id, task.model_dump(exclude_none=True)))
            if task.status.state in _TERMINAL_STATES:
                return
            if asyncio.get_running_loop().time() >= deadline:
                yield _sse(failure(envelope.id, INTERNAL_ERROR, "Timed out waiting for the task."))
                return
            await asyncio.sleep(_POLL_INTERVAL_SECS)

    return make_sse_response(sse_to_bytes(envelopes()), headers=get_sse_headers())


async def _tasks_get(_assistant_id: str, envelope: JSONRPCRequest, user: User) -> JSONResponse:
    params, error = _task_params(envelope)
    if params is None:
        return error

    task = await a2a_service.get_task(params.id, params.context_id, user)
    if task is None:
        return JSONResponse(failure(envelope.id, TASK_NOT_FOUND, "Task not found."))
    return JSONResponse(success(envelope.id, task.model_dump(exclude_none=True)))


async def _tasks_cancel(_assistant_id: str, envelope: JSONRPCRequest, _user: User) -> JSONResponse:
    return JSONResponse(
        failure(envelope.id, TASK_NOT_CANCELABLE, "tasks/cancel is not supported; cancel the run instead.")
    )


def _message_params(envelope: JSONRPCRequest) -> tuple[MessageParams | None, JSONResponse]:
    try:
        return MessageParams.model_validate(envelope.params or {}), JSONResponse({})
    except ValidationError as exc:
        message = str(exc.errors()[0].get("msg", "invalid params"))
        return None, JSONResponse(failure(envelope.id, INVALID_PARAMS, message))


def _task_params(envelope: JSONRPCRequest) -> tuple[TaskParams | None, JSONResponse]:
    try:
        return TaskParams.model_validate(envelope.params or {}), JSONResponse({})
    except ValidationError as exc:
        message = str(exc.errors()[0].get("msg", "invalid params"))
        return None, JSONResponse(failure(envelope.id, INVALID_PARAMS, message))


def _sse(envelope: dict[str, Any]) -> str:
    return f"data: {json.dumps(envelope)}\n\n"
