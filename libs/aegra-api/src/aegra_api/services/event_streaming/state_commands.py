"""Agent Protocol v2 state and agent commands.

``state.get``, ``state.listCheckpoints``, ``state.fork`` and ``agent.getTree``
read through the same LangGraph checkpointer the REST state endpoints use, so
both transports report identical state.
"""

from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.event_streaming.protocol import build_error, build_success
from aegra_api.services.langgraph_service import create_thread_config, get_langgraph_service

_RUN_STATUS_TO_AGENT_STATUS = {
    "pending": "started",
    "running": "running",
    "success": "completed",
    "error": "failed",
    "timeout": "failed",
    "interrupted": "interrupted",
}


def _namespace_to_ns(namespace: Any) -> str:
    if isinstance(namespace, list):
        return "|".join(str(part) for part in namespace)
    return ""


async def _thread_graph_id(session: AsyncSession, thread_id: str, user: User) -> str | None:
    metadata = await session.scalar(
        select(ThreadORM.metadata_json).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    )
    return (metadata or {}).get("graph_id") if isinstance(metadata, dict) else None


def _build_config(thread_id: str, user: User, namespace: Any, checkpoint_id: str | None = None) -> dict[str, Any]:
    config = create_thread_config(thread_id, user)
    ns = _namespace_to_ns(namespace)
    if ns:
        config["configurable"]["checkpoint_ns"] = ns
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    return config


async def state_get(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    graph_id = await _thread_graph_id(session, thread_id, user)
    if not graph_id:
        return build_success(command_id, {"values": {}}), None

    config = _build_config(thread_id, user, params.get("namespace"))
    async with get_langgraph_service().get_graph(
        graph_id, config=config, access_context="threads.read", user=user
    ) as agent:
        snapshot = await agent.aget_state(config)

    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    keys = params.get("keys")
    if isinstance(keys, list):
        values = {k: v for k, v in values.items() if k in keys}

    result: dict[str, Any] = {"values": jsonable_encoder(values)}
    configurable = (snapshot.config or {}).get("configurable") or {}
    if configurable.get("checkpoint_id"):
        result["checkpoint"] = {"id": configurable["checkpoint_id"], "ns": configurable.get("checkpoint_ns", "")}
    return build_success(command_id, result), None


async def state_list_checkpoints(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    graph_id = await _thread_graph_id(session, thread_id, user)
    if not graph_id:
        return build_success(command_id, {"checkpoints": []}), None

    limit = params.get("limit")
    limit = limit if isinstance(limit, int) and limit > 0 else 10
    config = _build_config(thread_id, user, params.get("namespace"))
    before = params.get("before")
    before_config = _build_config(thread_id, user, params.get("namespace"), before) if before else None

    checkpoints: list[dict[str, Any]] = []
    async with get_langgraph_service().get_graph(
        graph_id, config=config, access_context="threads.read", user=user
    ) as agent:
        async for snapshot in agent.aget_state_history(config, before=before_config, limit=limit):
            configurable = (snapshot.config or {}).get("configurable") or {}
            metadata = snapshot.metadata or {}
            entry: dict[str, Any] = {
                "id": configurable.get("checkpoint_id", ""),
                "timestamp": snapshot.created_at or "",
                "step": int(metadata.get("step", -1)),
            }
            if metadata.get("writes"):
                entry["node_name"] = next(iter(metadata["writes"]), None)
            entry["metadata"] = jsonable_encoder(metadata)
            checkpoints.append(entry)

    return build_success(command_id, {"checkpoints": checkpoints}), None


async def agent_get_tree(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    graph_id = await _thread_graph_id(session, thread_id, user)
    if not graph_id:
        return build_error(command_id, "no_such_run", "Thread has no associated graph."), None

    run_id = params.get("run_id")
    stmt = select(RunORM.status).where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
    if isinstance(run_id, str) and run_id:
        stmt = stmt.where(RunORM.run_id == run_id)
    status = await session.scalar(stmt.order_by(RunORM.created_at.desc()).limit(1))

    tree = {
        "namespace": [],
        "status": _RUN_STATUS_TO_AGENT_STATUS.get(status or "", "completed"),
        "graph_name": graph_id,
    }
    return build_success(command_id, {"tree": tree}), None


_SUBSCRIPTION_COMMANDS = frozenset({"subscription.subscribe", "subscription.unsubscribe", "subscription.reconnect"})


def subscription_not_supported(command_id: int, method: str) -> tuple[dict[str, Any], str | None]:
    """Channels are fixed per connection in the EventStreamRequest body."""
    return build_error(
        command_id,
        "not_supported",
        f"{method} is not supported; set channels in the /stream/events request body instead.",
    ), None


async def state_fork(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
    start: Any,
) -> tuple[dict[str, Any], str | None]:
    checkpoint_id = params.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        return build_error(command_id, "invalid_argument", "state.fork requires a string checkpoint_id."), None

    assistant_id = params.get("assistant_id") or await _thread_assistant_id(session, thread_id, user)
    if not assistant_id:
        return build_error(command_id, "no_such_run", "No run on this thread to fork from."), None

    request = RunCreate(
        assistant_id=assistant_id,
        input=params.get("input"),
        config=params.get("config") or {},
        checkpoint={"checkpoint_id": checkpoint_id},
    )
    run_id = await start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id, "thread_id": thread_id}), run_id


async def input_inject(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
    start: Any,
) -> tuple[dict[str, Any], str | None]:
    message = params.get("message")
    if not isinstance(message, dict) or not message.get("content"):
        return build_error(command_id, "invalid_argument", "input.inject requires a message with content."), None

    assistant_id = params.get("assistant_id") or await _thread_assistant_id(session, thread_id, user)
    if not assistant_id:
        return build_error(command_id, "no_such_run", "No run on this thread to inject into."), None

    request = RunCreate(
        assistant_id=assistant_id,
        input={"messages": [{"role": message.get("role", "user"), "content": message["content"]}]},
        config=params.get("config") or {},
    )
    run_id = await start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}), run_id


async def _thread_assistant_id(session: AsyncSession, thread_id: str, user: User) -> str | None:
    return await session.scalar(
        select(RunORM.assistant_id)
        .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )
