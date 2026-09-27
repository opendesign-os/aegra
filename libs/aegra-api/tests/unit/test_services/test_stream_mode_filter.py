"""`stream_mode` narrowing on the join stream, and the replay-buffer opt-in."""

from typing import Any

import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services import run_executor
from aegra_api.services.event_converter import EventConverter
from aegra_api.services.streaming_service import _CONTROL_MODES, streaming_service


def _job(*, stream_resumable: bool) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="t-1", graph_id="agent"),
        user=User(identity="test-user"),
        execution=RunExecution(stream_resumable=stream_resumable),
        behavior=RunBehavior(),
    )


class TestEventMode:
    def test_reads_the_mode_from_a_two_tuple(self) -> None:
        assert EventConverter().event_mode(("values", {"a": 1})) == "values"

    def test_reads_the_mode_from_a_subgraph_triple(self) -> None:
        converter = EventConverter()
        converter.set_subgraphs(True)

        assert converter.event_mode((["parent", "child"], "updates", {})) == "updates"

    def test_non_tuple_events_have_no_mode(self) -> None:
        assert EventConverter().event_mode({"not": "a tuple"}) is None


class TestAllowedFilter:
    def test_no_filter_lets_everything_through(self) -> None:
        assert streaming_service._allowed(("debug", {}), None) is True

    def test_requested_mode_passes(self) -> None:
        assert streaming_service._allowed(("values", {}), frozenset({"values"})) is True

    def test_unrequested_mode_is_dropped(self) -> None:
        assert streaming_service._allowed(("debug", {}), frozenset({"values"})) is False

    @pytest.mark.parametrize("mode", sorted(_CONTROL_MODES))
    def test_protocol_events_always_pass(self, mode: str) -> None:
        """Filtering out `end` would strip the terminal signal every client waits on."""
        assert streaming_service._allowed((mode, {}), frozenset({"values"})) is True

    def test_unparseable_events_are_kept(self) -> None:
        assert streaming_service._allowed("raw string", frozenset({"values"})) is True


class TestReplayEnabled:
    def test_request_opt_in_enables_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_executor.settings.event_streaming, "STREAM_RESUMABLE_ALWAYS", False)

        assert run_executor._replay_enabled(_job(stream_resumable=True)) is True

    def test_opted_out_request_skips_replay_in_upstream_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_executor.settings.event_streaming, "STREAM_RESUMABLE_ALWAYS", False)

        assert run_executor._replay_enabled(_job(stream_resumable=False)) is False

    def test_always_setting_preserves_aegra_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing deployments rely on reconnect working without opting in."""
        monkeypatch.setattr(run_executor.settings.event_streaming, "STREAM_RESUMABLE_ALWAYS", True)

        assert run_executor._replay_enabled(_job(stream_resumable=False)) is True


class TestPutToBrokerHonoursResumable:
    @pytest.mark.asyncio
    async def test_flag_reaches_the_broker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _Broker:
            async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
                captured.update(event_id=event_id, payload=payload, resumable=resumable)

        monkeypatch.setattr(
            "aegra_api.services.streaming_service.broker_manager.get_or_create_broker",
            lambda _run_id: _Broker(),
        )

        await streaming_service.put_to_broker("run-1", "run-1_event_1", ("values", {}), resumable=False)

        assert captured["resumable"] is False
