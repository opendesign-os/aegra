"""Terminal-run webhook delivery: signing, target restrictions, retries and scheduling."""

import asyncio
import base64
import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from aegra_api.services import webhook_delivery
from aegra_api.settings import settings
from aegra_api.utils import webhook as webhook_utils
from aegra_api.utils.webhook import mask_webhook_credentials, validate_webhook_url

PAYLOAD = {"run_id": "run-1", "thread_id": "t-1", "status": "success", "output": {"ok": True}}
URL = "https://hooks.example/cb"
SECRET = "whsec_" + base64.b64encode(b"super-secret-key").decode()


class _Recorder:
    """Stands in for httpx.AsyncClient, replaying a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.headers: list[dict[str, str]] = []
        self.bodies: list[bytes] = []

    def __call__(self, *_args: Any, **_kwargs: Any) -> "_Recorder":
        return self

    async def __aenter__(self) -> "_Recorder":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, content: bytes, headers: dict[str, str]) -> httpx.Response:
        self.calls.append((url, json.loads(content)))
        self.headers.append(headers)
        self.bodies.append(content)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=httpx.Request("POST", url))


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """tenacity waits between attempts; collapse it so retries run instantly."""
    monkeypatch.setattr(webhook_delivery, "wait_exponential", lambda **_kwargs: wait_none())


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every test host to a public address; no test should touch real DNS."""

    async def resolve(_host: str, _port: int) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(webhook_utils, "_resolve", resolve)


@pytest.fixture(autouse=True)
def _unsigned_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_SECRET", None)
    monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_ALLOW_PRIVATE_NETWORKS", False)
    monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_ALLOWED_HOSTS", "")


class TestDeliver:
    @pytest.mark.asyncio
    async def test_posts_the_payload_once_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, PAYLOAD) is True
        assert recorder.calls == [(URL, PAYLOAD)]

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([500, 200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, PAYLOAD) is True
        assert len(recorder.calls) == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_three_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([503])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, PAYLOAD) is False
        assert len(recorder.calls) == webhook_delivery.ATTEMPTS

    @pytest.mark.asyncio
    async def test_transport_errors_are_retried_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([httpx.ConnectError("refused"), 200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, PAYLOAD) is True
        assert len(recorder.calls) == 2

    @pytest.mark.asyncio
    async def test_unserializable_payload_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, {"output": object()}) is False
        assert recorder.calls == []


class TestSignature:
    """Standard Webhooks: HMAC-SHA256 over "{id}.{timestamp}.{body}"."""

    @pytest.mark.asyncio
    async def test_unsigned_when_no_secret_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        await webhook_delivery.deliver(URL, PAYLOAD)

        assert "webhook-signature" not in recorder.headers[0]
        assert recorder.headers[0]["webhook-id"]

    @pytest.mark.asyncio
    async def test_signature_verifies_against_the_sent_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_SECRET", SECRET)
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        await webhook_delivery.deliver(URL, PAYLOAD)

        headers, body = recorder.headers[0], recorder.bodies[0]
        signed = f"{headers['webhook-id']}.{headers['webhook-timestamp']}.".encode() + body
        expected = hmac.new(b"super-secret-key", signed, hashlib.sha256).digest()
        assert headers["webhook-signature"] == f"v1,{base64.b64encode(expected).decode()}"

    @pytest.mark.asyncio
    async def test_a_raw_secret_is_used_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_SECRET", "plain-key")
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        await webhook_delivery.deliver(URL, PAYLOAD)

        headers, body = recorder.headers[0], recorder.bodies[0]
        signed = f"{headers['webhook-id']}.{headers['webhook-timestamp']}.".encode() + body
        expected = hmac.new(b"plain-key", signed, hashlib.sha256).digest()
        assert headers["webhook-signature"] == f"v1,{base64.b64encode(expected).decode()}"

    @pytest.mark.asyncio
    async def test_the_id_is_stable_across_retries_so_receivers_can_deduplicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_SECRET", SECRET)
        recorder = _Recorder([500, 200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        await webhook_delivery.deliver(URL, PAYLOAD)

        assert recorder.headers[0]["webhook-id"] == recorder.headers[1]["webhook-id"]


class TestTargetGuard:
    """Webhook URLs come from callers, so the delivery target is restricted."""

    @pytest.mark.asyncio
    async def test_a_host_resolving_inside_the_network_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def resolve(_host: str, _port: int) -> list[str]:
            return ["169.254.169.254"]

        monkeypatch.setattr(webhook_utils, "_resolve", resolve)
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("https://metadata.example/cb", PAYLOAD) is False
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_a_named_host_is_allowed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def resolve(_host: str, _port: int) -> list[str]:
            return ["10.0.0.7"]

        monkeypatch.setattr(webhook_utils, "_resolve", resolve)
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_ALLOWED_HOSTS", "worker.internal, other.internal")
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("http://worker.internal/cb", PAYLOAD) is True

    @pytest.mark.asyncio
    async def test_the_guard_can_be_dropped_wholesale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_ALLOW_PRIVATE_NETWORKS", True)
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver("http://127.0.0.1:9000/cb", PAYLOAD) is True

    @pytest.mark.asyncio
    async def test_a_host_that_does_not_resolve_still_gets_its_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def resolve(_host: str, _port: int) -> list[str]:
            raise OSError("Name or service not known")

        monkeypatch.setattr(webhook_utils, "_resolve", resolve)
        recorder = _Recorder([httpx.ConnectError("no such host")])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        assert await webhook_delivery.deliver(URL, PAYLOAD) is False
        assert len(recorder.calls) == webhook_delivery.ATTEMPTS


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

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:9000/cb",
            "http://10.0.0.7/cb",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:9000/cb",
            "http://localhost:3000/cb",
            "http://api.localhost/cb",
        ],
    )
    def test_a_literal_internal_address_is_refused_at_request_time(self, url: str) -> None:
        with pytest.raises(ValueError, match="non-public address"):
            validate_webhook_url(url)

    def test_a_literal_public_address_is_accepted(self) -> None:
        assert validate_webhook_url("https://93.184.216.34/cb") == "https://93.184.216.34/cb"

    def test_a_listed_literal_address_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.webhook, "AEGRA_WEBHOOK_ALLOWED_HOSTS", "10.0.0.7")

        assert validate_webhook_url("http://10.0.0.7/cb") == "http://10.0.0.7/cb"


class TestSchedule:
    @pytest.mark.asyncio
    async def test_no_url_schedules_nothing(self) -> None:
        webhook_delivery.schedule(None, PAYLOAD)

        assert webhook_delivery._pending == set()

    @pytest.mark.asyncio
    async def test_delivery_runs_without_blocking_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder([200])
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", recorder)

        webhook_delivery.schedule(URL, PAYLOAD)
        assert recorder.calls == []

        await asyncio.sleep(0)
        await asyncio.gather(*list(webhook_delivery._pending))

        assert len(recorder.calls) == 1

    @pytest.mark.asyncio
    async def test_finished_tasks_are_released(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pending set must not grow unbounded across runs."""
        monkeypatch.setattr(webhook_delivery.httpx, "AsyncClient", _Recorder([200]))

        webhook_delivery.schedule(URL, PAYLOAD)
        await asyncio.gather(*list(webhook_delivery._pending))
        await asyncio.sleep(0)

        assert webhook_delivery._pending == set()
