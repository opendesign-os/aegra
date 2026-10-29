"""Map A2A tasks onto Aegra runs.

Models, errors and request envelopes all come from ``a2a.compat.v0_3.types`` —
the SDK's JSON-RPC shapes. ``message.contextId`` is the LangGraph thread and an
Aegra run id is the A2A task id.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from a2a.compat.v0_3 import types as a2a
from sqlalchemy import select

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.executor import executor
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

TERMINAL_STATES = frozenset(
    {a2a.TaskState.completed, a2a.TaskState.failed, a2a.TaskState.canceled, a2a.TaskState.rejected}
)

_RUN_STATUS_TO_TASK_STATE = {
    "pending": a2a.TaskState.submitted,
    "running": a2a.TaskState.working,
    "success": a2a.TaskState.completed,
    "error": a2a.TaskState.failed,
    "timeout": a2a.TaskState.failed,
    "interrupted": a2a.TaskState.input_required,
}


def unsupported_part_kinds(message: a2a.Message) -> list[str]:
    """Part kinds this deployment cannot carry.

    The A2A spec allows file parts; this endpoint does not, so they are
    reported rather than dropped on the floor.
    """
    return sorted(
        {
            getattr(part.root, "kind", "unknown")
            for part in message.parts
            if not isinstance(part.root, (a2a.TextPart, a2a.DataPart))
        }
    )


def message_to_input(message: a2a.Message) -> dict[str, Any]:
    """Fold the A2A parts into a graph input payload."""
    payload: dict[str, Any] = {}
    messages: list[dict[str, str]] = []
    role = "user" if message.role == a2a.Role.user else "assistant"

    for part in message.parts:
        root = part.root
        if isinstance(root, a2a.TextPart):
            messages.append({"role": role, "content": root.text})
        elif isinstance(root, a2a.DataPart):
            payload.update(root.data)

    if messages:
        payload["messages"] = messages
    return payload


def run_to_task(run: RunORM) -> a2a.Task:
    state = _RUN_STATUS_TO_TASK_STATE.get(run.status, a2a.TaskState.working)
    artifacts = None
    if run.output:
        artifacts = [
            a2a.Artifact(
                artifact_id=run.run_id,
                parts=[a2a.Part(root=a2a.DataPart(data=run.output))],
            )
        ]

    status_message = None
    if run.error_message:
        status_message = a2a.Message(
            role=a2a.Role.agent,
            parts=[a2a.Part(root=a2a.TextPart(text=run.error_message))],
            message_id=run.run_id,
            context_id=run.thread_id,
            task_id=run.run_id,
        )

    return a2a.Task(
        id=run.run_id,
        context_id=run.thread_id,
        status=a2a.TaskStatus(
            state=state,
            timestamp=(run.updated_at or datetime.now(UTC)).isoformat(),
            message=status_message,
        ),
        artifacts=artifacts,
    )


async def start_task(assistant_id: str, message: a2a.Message, user: User) -> tuple[str, str]:
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


async def wait_for_task(task_id: str, thread_id: str, user: User) -> a2a.Task:
    with contextlib.suppress(TimeoutError):
        await executor.wait_for_completion(task_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
    task = await get_task(task_id, thread_id, user)
    if task is None:
        raise RuntimeError(f"Run {task_id} vanished while the A2A task was waiting")
    return task


async def get_task(task_id: str, thread_id: str, user: User) -> a2a.Task | None:
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
