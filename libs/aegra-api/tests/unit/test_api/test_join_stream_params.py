"""`stream_mode` and `cancel_on_disconnect` on the reconnect-style join routes.

The SDK sends both on `runs.join_stream` and `runs.join`. Before these landed the
server declared a differently-named parameter, so the values were silently dropped.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.api.runs import join_run, stream_run
from aegra_api.core.orm import Run as RunORM
from aegra_api.models.auth import User


def _run_row(status: str = "running") -> RunORM:
    now = datetime.now(UTC)
    return RunORM(
        run_id="run-1",
        thread_id="t-1",
        assistant_id="agent",
        status=status,
        input={},
        config={},
        context={},
        user_id="test-user",
        created_at=now,
        updated_at=now,
    )


def _session_maker(row: Any) -> MagicMock:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=row)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


async def _fake_stream() -> Any:
    yield "data"


@pytest.fixture
def user() -> User:
    return User(identity="test-user")


class TestStreamModeFilter:
    @pytest.mark.asyncio
    async def test_requested_modes_reach_the_producer(self, user: User) -> None:
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ) as producer,
        ):
            await stream_run(
                "t-1",
                "run-1",
                last_event_id=None,
                stream_mode=["values", "updates"],
                cancel_on_disconnect=False,
                user=user,
            )

        assert producer.call_args.args[2] == ["values", "updates"]

    @pytest.mark.asyncio
    async def test_omitted_modes_leave_the_stream_unfiltered(self, user: User) -> None:
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ) as producer,
        ):
            await stream_run(
                "t-1", "run-1", last_event_id=None, stream_mode=None, cancel_on_disconnect=False, user=user
            )

        assert producer.call_args.args[2] is None


class TestCancelOnDisconnect:
    @pytest.mark.asyncio
    async def test_opting_in_wires_a_close_handler(self, user: User) -> None:
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ),
        ):
            response = await stream_run(
                "t-1", "run-1", last_event_id=None, stream_mode=None, cancel_on_disconnect=True, user=user
            )

        assert response.client_close_handler_callable is not None

    @pytest.mark.asyncio
    async def test_default_leaves_a_shared_run_alone(self, user: User) -> None:
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ),
        ):
            response = await stream_run(
                "t-1", "run-1", last_event_id=None, stream_mode=None, cancel_on_disconnect=False, user=user
            )

        assert response.client_close_handler_callable is None

    @pytest.mark.asyncio
    async def test_handler_requests_cancellation(self, user: User) -> None:
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ),
            patch("aegra_api.api.runs.broker_manager.request_cancel", AsyncMock()) as cancel,
        ):
            response = await stream_run(
                "t-1", "run-1", last_event_id=None, stream_mode=None, cancel_on_disconnect=True, user=user
            )
            await response.client_close_handler_callable({})

        cancel.assert_awaited_once_with("run-1", "cancel")

    @pytest.mark.asyncio
    async def test_infrastructure_failures_do_not_escape_the_handler(self, user: User) -> None:
        """A raising close handler would break sse-starlette's teardown."""
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch(
                "aegra_api.api.runs.streaming_service.stream_run_execution",
                return_value=_fake_stream(),
            ),
            patch(
                "aegra_api.api.runs.broker_manager.request_cancel",
                AsyncMock(side_effect=ConnectionError("redis gone")),
            ),
        ):
            response = await stream_run(
                "t-1", "run-1", last_event_id=None, stream_mode=None, cancel_on_disconnect=True, user=user
            )

            await response.client_close_handler_callable({})


class TestJoinCancelOnDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_probe_is_passed_when_opted_in(self, user: User) -> None:
        request = MagicMock()
        request.is_disconnected = AsyncMock(return_value=False)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch("aegra_api.api.runs.heartbeat_wait_body", return_value=_fake_stream()) as body,
        ):
            await join_run("t-1", "run-1", request, cancel_on_disconnect=True, user=user)

        assert body.call_args.kwargs["is_disconnected"] is request.is_disconnected

    @pytest.mark.asyncio
    async def test_no_probe_by_default(self, user: User) -> None:
        request = MagicMock()

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=_session_maker(_run_row())),
            patch("aegra_api.api.runs.heartbeat_wait_body", return_value=_fake_stream()) as body,
        ):
            await join_run("t-1", "run-1", request, cancel_on_disconnect=False, user=user)

        assert body.call_args.kwargs["is_disconnected"] is None


class TestHeartbeatDisconnectPolling:
    """The probe runs on the heartbeat tick, not in teardown, so the cancel gets a live loop."""

    async def _drain(self, **kwargs: Any) -> list[bytes]:
        from aegra_api.services.run_waiters import heartbeat_wait_body

        chunks: list[bytes] = []
        async for chunk in heartbeat_wait_body("run-1", "t-1", "test-user", timeout=5, **kwargs):
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_disconnected_client_cancels_the_run(self) -> None:
        with (
            patch("aegra_api.services.run_waiters.settings.app.KEEPALIVE_INTERVAL_SECS", 0.01),
            patch("aegra_api.services.run_waiters.executor.wait_for_completion", AsyncMock(side_effect=_never)),
            patch("aegra_api.services.run_waiters.broker_manager.request_cancel", AsyncMock()) as cancel,
        ):
            chunks = await self._drain(is_disconnected=AsyncMock(return_value=True))

        cancel.assert_awaited_once_with("run-1", "cancel")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_connected_client_keeps_receiving_heartbeats(self) -> None:
        probe = AsyncMock(side_effect=[False, False, True])

        with (
            patch("aegra_api.services.run_waiters.settings.app.KEEPALIVE_INTERVAL_SECS", 0.01),
            patch("aegra_api.services.run_waiters.executor.wait_for_completion", AsyncMock(side_effect=_never)),
            patch("aegra_api.services.run_waiters.broker_manager.request_cancel", AsyncMock()),
        ):
            chunks = await self._drain(is_disconnected=probe)

        assert chunks == [b"\n", b"\n"]

    @pytest.mark.asyncio
    async def test_without_the_probe_the_run_is_never_cancelled(self) -> None:
        with (
            patch("aegra_api.services.run_waiters.settings.app.KEEPALIVE_INTERVAL_SECS", 0.01),
            patch("aegra_api.services.run_waiters.executor.wait_for_completion", AsyncMock(return_value=None)),
            patch("aegra_api.services.run_waiters.read_run_result", AsyncMock(return_value={"output": {}})),
            patch("aegra_api.services.run_waiters.broker_manager.request_cancel", AsyncMock()) as cancel,
        ):
            await self._drain()

        cancel.assert_not_awaited()


async def _never(*_args: Any, **_kwargs: Any) -> None:
    import asyncio

    await asyncio.sleep(3600)
