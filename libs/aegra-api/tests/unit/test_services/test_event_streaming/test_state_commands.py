"""v2 state and agent commands: state.get, state.listCheckpoints, agent.getTree, fork, inject."""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from aegra_api.models import User
from aegra_api.services.event_streaming import commands as cmd
from aegra_api.services.event_streaming import state_commands as sc


@pytest.fixture
def user() -> User:
    return User(identity="u1")


@pytest.fixture
def prepared_run(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=("run-xyz", object(), object()))
    monkeypatch.setattr(cmd, "_prepare_run", mock)
    return mock


class _Snapshot:
    def __init__(self, values: Any, config: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None):
        self.values = values
        self.config = config or {"configurable": {"checkpoint_id": "cp-1", "checkpoint_ns": ""}}
        self.metadata = metadata or {"step": 3, "writes": {"agent": {}}}
        self.created_at = "2026-01-01T00:00:00+00:00"
        self.tasks: list[Any] = []


def _graph(snapshot: _Snapshot, history: list[_Snapshot] | None = None) -> Any:
    class _Agent:
        async def aget_state(self, _config: dict[str, Any]) -> _Snapshot:
            return snapshot

        async def aget_state_history(self, _config: dict[str, Any], **_kw: Any) -> Any:
            for item in history or []:
                yield item

    @asynccontextmanager
    async def _get_graph(*_args: Any, **_kwargs: Any) -> Any:
        yield _Agent()

    service = AsyncMock()
    service.get_graph = _get_graph
    return service


def _session(*, graph_id: str | None = "agent", scalar: Any = None) -> AsyncMock:
    session = AsyncMock()
    session.scalar = AsyncMock(
        return_value=scalar if scalar is not None else ({"graph_id": graph_id} if graph_id else None)
    )
    return session


async def _dispatch(payload: dict[str, Any], user: User, session: Any) -> tuple[dict, str | None]:
    return await cmd.handle_command(payload, session=session, thread_id="t1", user=user)


class TestStateGet:
    @pytest.mark.asyncio
    async def test_returns_values_and_checkpoint(self, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
        monkeypatch.setattr(sc, "get_langgraph_service", lambda: _graph(_Snapshot({"a": 1})))

        resp, run_id = await _dispatch({"id": 1, "method": "state.get", "params": {}}, user, _session())

        assert resp["type"] == "success"
        assert resp["result"]["values"] == {"a": 1}
        assert resp["result"]["checkpoint"] == {"id": "cp-1", "ns": ""}
        assert run_id is None

    @pytest.mark.asyncio
    async def test_keys_narrow_the_projection(self, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
        monkeypatch.setattr(sc, "get_langgraph_service", lambda: _graph(_Snapshot({"a": 1, "b": 2})))

        resp, _ = await _dispatch({"id": 1, "method": "state.get", "params": {"keys": ["b"]}}, user, _session())

        assert resp["result"]["values"] == {"b": 2}

    @pytest.mark.asyncio
    async def test_messages_are_json_encoded(self, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
        """LangChain messages are not JSON-serializable, and this envelope is dumped by stdlib json."""
        snapshot = _Snapshot({"messages": [HumanMessage(content="hi")]})
        monkeypatch.setattr(sc, "get_langgraph_service", lambda: _graph(snapshot))

        resp, _ = await _dispatch({"id": 1, "method": "state.get", "params": {}}, user, _session())

        message = resp["result"]["values"]["messages"][0]
        assert isinstance(message, dict)
        assert message["content"] == "hi"

    @pytest.mark.asyncio
    async def test_thread_without_graph_reports_empty_state(self, user: User) -> None:
        resp, _ = await _dispatch({"id": 1, "method": "state.get", "params": {}}, user, _session(graph_id=None))

        assert resp["result"] == {"values": {}}


class TestListCheckpoints:
    @pytest.mark.asyncio
    async def test_summarises_history(self, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
        history = [_Snapshot({}, metadata={"step": 2, "writes": {"agent": {}}})]
        monkeypatch.setattr(sc, "get_langgraph_service", lambda: _graph(_Snapshot({}), history))

        resp, _ = await _dispatch(
            {"id": 1, "method": "state.listCheckpoints", "params": {"limit": 5}}, user, _session()
        )

        checkpoint = resp["result"]["checkpoints"][0]
        assert checkpoint["id"] == "cp-1"
        assert checkpoint["step"] == 2
        assert checkpoint["node_name"] == "agent"

    @pytest.mark.asyncio
    async def test_thread_without_graph_has_no_checkpoints(self, user: User) -> None:
        resp, _ = await _dispatch(
            {"id": 1, "method": "state.listCheckpoints", "params": {}}, user, _session(graph_id=None)
        )

        assert resp["result"] == {"checkpoints": []}


class TestAgentGetTree:
    @pytest.mark.asyncio
    async def test_maps_run_status_to_agent_status(self, user: User) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[{"graph_id": "agent"}, "running"])

        resp, _ = await _dispatch({"id": 1, "method": "agent.getTree", "params": {}}, user, session)

        assert resp["result"]["tree"] == {"namespace": [], "status": "running", "graph_name": "agent"}

    @pytest.mark.asyncio
    async def test_thread_without_graph_is_an_error(self, user: User) -> None:
        resp, _ = await _dispatch({"id": 1, "method": "agent.getTree", "params": {}}, user, _session(graph_id=None))

        assert resp["type"] == "error"
        assert resp["error"] == "no_such_run"


class TestStateFork:
    @pytest.mark.asyncio
    async def test_starts_a_run_from_the_checkpoint(self, prepared_run: AsyncMock, user: User) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value="agent")

        resp, run_id = await _dispatch(
            {"id": 1, "method": "state.fork", "params": {"checkpoint_id": "cp-1"}}, user, session
        )

        assert resp["result"] == {"run_id": "run-xyz", "thread_id": "t1"}
        assert run_id == "run-xyz"
        assert prepared_run.call_args.args[2].checkpoint == {"checkpoint_id": "cp-1"}

    @pytest.mark.asyncio
    async def test_requires_a_checkpoint_id(self, user: User) -> None:
        resp, _ = await _dispatch({"id": 1, "method": "state.fork", "params": {}}, user, AsyncMock())

        assert resp["error"] == "invalid_argument"

    @pytest.mark.asyncio
    async def test_thread_without_runs_cannot_fork(self, user: User) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)

        resp, _ = await _dispatch({"id": 1, "method": "state.fork", "params": {"checkpoint_id": "cp-1"}}, user, session)

        assert resp["error"] == "no_such_run"


class TestInputInject:
    @pytest.mark.asyncio
    async def test_starts_a_run_carrying_the_message(self, prepared_run: AsyncMock, user: User) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value="agent")

        resp, run_id = await _dispatch(
            {"id": 1, "method": "input.inject", "params": {"message": {"role": "user", "content": "hi"}}},
            user,
            session,
        )

        assert resp["result"] == {"run_id": "run-xyz"}
        assert run_id == "run-xyz"
        assert prepared_run.call_args.args[2].input == {"messages": [{"role": "user", "content": "hi"}]}

    @pytest.mark.asyncio
    async def test_requires_message_content(self, user: User) -> None:
        resp, _ = await _dispatch({"id": 1, "method": "input.inject", "params": {"message": {}}}, user, AsyncMock())

        assert resp["error"] == "invalid_argument"


class TestSubscriptionCommands:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["subscription.subscribe", "subscription.unsubscribe", "subscription.reconnect"])
    async def test_answer_not_supported_rather_than_unknown(self, method: str, user: User) -> None:
        """They are spec commands we know about, so `unknown_command` would misreport them."""
        resp, _ = await _dispatch({"id": 1, "method": method, "params": {}}, user, AsyncMock())

        assert resp["type"] == "error"
        assert resp["error"] == "not_supported"
