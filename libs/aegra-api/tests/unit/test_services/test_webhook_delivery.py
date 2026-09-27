"""Terminal-run webhook delivery: retries, redaction, and fire-and-forget scheduling."""

import asyncio
from typing import Any

import httpx
import pytest

from aegra_api.services import webhook_delivery
from aegra_api.utils.webhook import mask_webhook_credentials, validate_webhook_url

PAYLOAD = {"run_id": "run-1", "thread_id": "t-1", "status": "success", "output": {"ok": True}}


class _Recorder:
    """Stands in for httpx.AsyncClient, replaying a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, *_args: Any, **_kwargs: Any) -> "_Recorder":
        return self

    async def __aenter__(self) -> "_Recorder":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, json: dict[str, Any]) -> httpx.Response:
        self.calls.append((url, json))
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=httpx.Request("POST", url))


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(webhook_delivery.asyncio, "sleep", _instant)


class TestDeliver:
    @pytest.mark.asyncio
    async def test_posts_the_payload_once_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("https://hooks.example/cb", PAYLOAD) is True
        assert recorder.calls == [("https://hooks.example/cb", PAYLOAD)]

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([500, 200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("https://hooks.example/cb", PAYLOAD) is True
        assert len(recorder.calls) == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_three_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([503])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("https://hooks.example/cb", PAYLOAD) is False
        assert len(recorder.calls) == webhook_delivery._ATTEMPTS

    @pytest.mark.asyncio
    async def test_transport_errors_are_retried_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([httpx.ConnectError("refused"), 200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("https://hooks.example/cb", PAYLOAD) is True
        assert len(recorder.calls) == 2


class TestSchedule:
    @pytest.mark.asyncio
    async def test_no_url_schedules_nothing(self) -> None:
        webhook_delivery.schedule(None, PAYLOAD)

        assert webhook_delivery._pending == set()

    @pytest.mark.asyncio
    async def test_delivery_runs_without_blocking_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        webhook_delivery.schedule("https://hooks.example/cb", PAYLOAD)
        assert recorder.calls == []

        await asyncio.sleep(0)
        await asyncio.gather(*list(webhook_delivery._pending))

        assert len(recorder.calls) == 1

    @pytest.mark.asyncio
    async def test_finished_tasks_are_released(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pending set must not grow unbounded across runs."""
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", _Recorder([200]))

        webhook_delivery.schedule("https://hooks.example/cb", PAYLOAD)
        await asyncio.gather(*list(webhook_delivery._pending))
        await asyncio.sleep(0)

        assert webhook_delivery._pending == set()


class TestUrlHandling:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://user:token@host.example/cb", "https://host.example/cb"),
            ("https://user:token@host.example:8443/cb", "https://host.example:8443/cb"),
            ("https://host.example/cb", "https://host.example/cb"),
        ],
    )
    def test_credentials_are_stripped_before_logging(self, url: str, expected: str) -> None:
        assert mask_webhook_credentials(url) == expected

    def test_validation_accepts_http_and_https(self) -> None:
        assert validate_webhook_url("http://host.example/cb") == "http://host.example/cb"

    def test_validation_passes_none_through(self) -> None:
        assert validate_webhook_url(None) is None
