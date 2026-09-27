"""Dispatch A2A JSON-RPC methods onto Aegra runs.

`message.contextId` is the LangGraph thread; an Aegra run id is the A2A task id.
Only text and data parts are carried: text becomes a `messages` entry, data is
merged into the graph input.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import User
from aegra_api.models.a2a import (
    A2AMessage,
    DataPart,
    Task,
    TaskStatus,
    TextPart,
)
from aegra_api.models.runs import RunCreate
from aegra_api.services.executor import executor
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_RUN_STATUS_TO_TASK_STATE = {
    "pending": "submitted",
    "running": "working",
    "success": "completed",
    "error": "failed",
    "timeout": "failed",
    "interrupted": "input-required",
}


def message_to_input(message: A2AMessage) -> dict[str, Any]:
    """Fold the A2A parts into a graph input payload."""
    payload: dict[str, Any] = {}
    messages: list[dict[str, str]] = []
    role = "user" if message.role == "user" else "assistant"

    for part in message.parts:
        if isinstance(part, TextPart):
            messages.append({"role": role, "content": part.text})
        elif isinstance(part, DataPart):
            payload.update(part.data)

    if messages:
        payload["messages"] = messages
    return payload


def run_to_task(run: RunORM) -> Task:
    state = _RUN_STATUS_TO_TASK_STATE.get(run.status, "working")
    artifacts = None
    if run.output:
        artifacts = [{"artifactId": run.run_id, "parts": [{"kind": "data", "data": run.output}]}]

    return Task(
        id=run.run_id,
        context_id=run.thread_id,
        status=TaskStatus(
            state=state,
            timestamp=(run.updated_at or datetime.now(UTC)).isoformat(),
            message={"error": run.error_message} if run.error_message else None,
        ),
        artifacts=artifacts,
    )


async def start_task(assistant_id: str, message: A2AMessage, user: User) -> tuple[str, str]:
    """Create the run backing an A2A task; returns ``(task_id, context_id)``."""
    thread_id = message.context_id or str(uuid4())
    maker = _get_session_maker()
    async with maker() as session:
        run_id, _run, _job = await _prepare_run(
            session,
            thread_id,
            RunCreate(assistant_id=assistant_id, input=message_to_input(message) or {}),
            user,
            initial_status="pending",
        )
    return run_id, thread_id


async def wait_for_task(task_id: str, thread_id: str, user: User) -> Task:
    with contextlib.suppress(TimeoutError):
        await executor.wait_for_completion(task_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
    task = await get_task(task_id, thread_id, user)
    if task is None:
        raise RuntimeError(f"Run {task_id} vanished while the A2A task was waiting")
    return task


async def get_task(task_id: str, thread_id: str, user: User) -> Task | None:
    maker = _get_session_maker()
    async with maker() as session:
        run = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == task_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
            )
        )
    return run_to_task(run) if run else None


async def thread_is_foreign(thread_id: str, user: User) -> bool:
    """True when the context belongs to somebody else."""
    maker = _get_session_maker()
    async with maker() as session:
        owner = await session.scalar(select(ThreadORM.user_id).where(ThreadORM.thread_id == thread_id))
    return owner is not None and owner != user.identity
