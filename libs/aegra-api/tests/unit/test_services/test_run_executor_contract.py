"""`webhook`, `after_seconds` and `durability` must reach the runtime, not stop at the model."""

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.run_executor import execute_run


async def _empty_async_gen():  # type: ignore[no-untyped-def]
    return
    yield  # noqa: RET504 — makes this an async generator


def _job(
    *,
    webhook: str | None = None,
    after_seconds: int | None = None,
    durability: str | None = None,
) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hi"}, durability=durability),
        behavior=RunBehavior(webhook=webhook, after_seconds=after_seconds),
    )


class _Harness(ExitStack):
    """Patches every collaborator execute_run touches, exposing the interesting mocks."""

    def _patch(self, target: str, *args: Any, **kwargs: Any) -> Any:
        return self.enter_context(patch(f"aegra_api.services.run_executor.{target}", *args, **kwargs))

    def __enter__(self) -> "_Harness":
        super().__enter__()
        graph = MagicMock()
        graph.__aenter__ = AsyncMock(return_value=graph)
        graph.__aexit__ = AsyncMock(return_value=False)
        service = MagicMock()
        service.get_graph = MagicMock(return_value=graph)

        auth_ctx = AsyncMock()
        auth_ctx.__aenter__ = AsyncMock(return_value=None)
        auth_ctx.__aexit__ = AsyncMock(return_value=False)

        p = self._patch

        p("get_langgraph_service", return_value=service)
        p("start_run", AsyncMock(return_value=True))
        self.finalize = p("finalize_run", AsyncMock(return_value=True))
        streaming = p("streaming_service")
        streaming.cleanup_run = AsyncMock()
        p("_signal_end_event", new_callable=AsyncMock)
        p("_signal_run_done", new_callable=AsyncMock)
        p("with_auth_ctx", return_value=auth_ctx)
        self.stream = p("stream_graph_events", return_value=_empty_async_gen())
        self.webhook = p("webhook_delivery")
        self.sleep = p("asyncio.sleep", new_callable=AsyncMock)
        return self


class TestWebhookOnCompletion:
    @pytest.mark.asyncio
    async def test_terminal_run_notifies_the_webhook(self) -> None:
        with _Harness() as harness:
            await execute_run(_job(webhook="https://hooks.example/cb"))

        harness.webhook.schedule.assert_called_once()
        url, payload = harness.webhook.schedule.call_args.args
        assert url == "https://hooks.example/cb"
        assert payload["run_id"] == "run-1"
        assert payload["status"] == "success"

    @pytest.mark.asyncio
    async def test_no_webhook_still_schedules_nothing_harmful(self) -> None:
        """schedule() is called with None and short-circuits, keeping one call site."""
        with _Harness() as harness:
            await execute_run(_job())

        url, _payload = harness.webhook.schedule.call_args.args
        assert url is None

    @pytest.mark.asyncio
    async def test_a_run_that_was_already_terminal_does_not_notify(self) -> None:
        with _Harness() as harness:
            harness.finalize.return_value = False

            await execute_run(_job(webhook="https://hooks.example/cb"))

        harness.webhook.schedule.assert_not_called()


class TestAfterSeconds:
    @pytest.mark.asyncio
    async def test_delay_is_awaited_before_the_run_starts(self) -> None:
        with _Harness() as harness:
            await execute_run(_job(after_seconds=30))

        harness.sleep.assert_awaited_once_with(30)

    @pytest.mark.asyncio
    async def test_no_delay_when_unset(self) -> None:
        with _Harness() as harness:
            await execute_run(_job())

        harness.sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_zero_is_not_a_delay(self) -> None:
        with _Harness() as harness:
            await execute_run(_job(after_seconds=0))

        harness.sleep.assert_not_awaited()


class TestDurabilityReachesPregel:
    @pytest.mark.asyncio
    async def test_value_is_forwarded_to_the_stream(self) -> None:
        with _Harness() as harness:
            await execute_run(_job(durability="exit"))

        assert harness.stream.call_args.kwargs["durability"] == "exit"

    @pytest.mark.asyncio
    async def test_unset_leaves_the_runtime_default(self) -> None:
        with _Harness() as harness:
            await execute_run(_job())

        assert harness.stream.call_args.kwargs["durability"] is None


class TestWebhookPayloadShape:
    @pytest.mark.asyncio
    async def test_carries_the_fields_a_receiver_needs(self) -> None:
        with _Harness() as harness:
            await execute_run(_job(webhook="https://hooks.example/cb"))

        _url, payload = harness.webhook.schedule.call_args.args
        assert set(payload) == {"run_id", "thread_id", "assistant_id", "status", "output"}
