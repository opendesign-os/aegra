"""HTTP-level tests for the endpoints added to match the LangSmith Deployment contract."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummyScalarResult, DummySessionBase, override_get_session_dep
from tests.fixtures.session_fixtures import override_session_dependency
from tests.fixtures.test_helpers import DummyThread


@pytest.fixture
def client() -> TestClient:
    app = create_test_app()
    override_session_dependency(app)
    return make_client(app)


def _run_model(run_id: str) -> Any:
    from datetime import UTC, datetime

    from aegra_api.models.runs import Run

    now = datetime.now(UTC)
    return Run(
        run_id=run_id,
        thread_id="t-1",
        assistant_id="agent",
        status="pending",
        user_id="test-user",
        created_at=now,
        updated_at=now,
    )


class TestOk:
    """`GET /ok` is the liveness name the platform advertises."""

    def test_reports_ok(self, client: TestClient) -> None:
        response = client.get("/ok")

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_check_db_zero_skips_the_database(self, client: TestClient) -> None:
        response = client.get("/ok", params={"check_db": 0})

        assert response.status_code == 200

    def test_check_db_one_probes_the_database(self, client: TestClient) -> None:
        with patch("aegra_api.core.health.db_manager") as manager:
            connection = AsyncMock()
            manager.get_engine.return_value.connect.return_value.__aenter__.return_value = connection

            response = client.get("/ok", params={"check_db": 1})

        assert response.status_code == 200
        connection.execute.assert_awaited()

    def test_rejects_out_of_range_check_db(self, client: TestClient) -> None:
        assert client.get("/ok", params={"check_db": 2}).status_code == 422


class TestMetrics:
    """`/metrics` is served by the Prometheus instrumentator, gated on its setting."""

    def _client(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        import prometheus_client

        from aegra_api.observability import metrics as metrics_module

        monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", enabled)
        app = create_test_app()
        metrics_module.setup_prometheus_metrics(app, registry=prometheus_client.CollectorRegistry())
        return make_client(app)

    def test_disabled_by_default_leaves_no_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The endpoint is deliberately unauthenticated, so it stays opt-in."""
        assert self._client(False, monkeypatch).get("/metrics").status_code == 404

    def test_defaults_to_prometheus_exposition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = self._client(True, monkeypatch).get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "# HELP" in response.text

    def test_format_json_returns_an_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = self._client(True, monkeypatch).get("/metrics", params={"format": "json"})

        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_rejects_unknown_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._client(True, monkeypatch).get("/metrics", params={"format": "csv"}).status_code == 422


class _CountSession(DummySessionBase):
    """Session whose scalar() answers the count query."""

    def __init__(self, total: int = 7) -> None:
        super().__init__()
        self.total = total
        self.statements: list[Any] = []

    async def scalar(self, stmt: Any = None) -> Any:
        self.statements.append(stmt)
        return self.total


class TestThreadCount:
    def _client(self, session: _CountSession) -> TestClient:
        app = create_test_app()
        from aegra_api.core.orm import get_session as core_get_session

        app.dependency_overrides[core_get_session] = override_get_session_dep(lambda: session)
        return make_client(app)

    def test_returns_an_integer(self) -> None:
        response = self._client(_CountSession(7)).post("/threads/count", json={})

        assert response.status_code == 200
        assert response.json() == 7

    def test_accepts_the_spec_filters(self) -> None:
        response = self._client(_CountSession(2)).post("/threads/count", json={"status": "idle", "metadata": {"a": 1}})

        assert response.status_code == 200

    def test_rejects_unknown_status(self) -> None:
        response = self._client(_CountSession()).post("/threads/count", json={"status": "sleeping"})

        assert response.status_code == 422


class _CopySession(DummySessionBase):
    def __init__(self, source: Any) -> None:
        super().__init__()
        self.source = source
        self.added: list[Any] = []

    async def scalar(self, stmt: Any = None) -> Any:
        return self.source

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        return None


class TestThreadCopy:
    def _client(self, session: _CopySession) -> TestClient:
        app = create_test_app()
        from aegra_api.core.orm import get_session as core_get_session

        app.dependency_overrides[core_get_session] = override_get_session_dep(lambda: session)
        return make_client(app)

    def test_forks_into_a_new_thread_id(self) -> None:
        source = DummyThread("src-thread", "idle", {"graph_id": "agent"}, "test-user")
        source.metadata_json = {"graph_id": "agent"}
        session = _CopySession(source)

        with (
            patch("aegra_api.api.threads.copy_thread_checkpoints", AsyncMock()) as copier,
            patch("aegra_api.api.threads.db_manager"),
        ):
            response = self._client(session).post("/threads/src-thread/copy")

        assert response.status_code == 200
        assert response.json()["thread_id"] != "src-thread"
        copier.assert_awaited_once()

    def test_carries_the_source_metadata(self) -> None:
        source = DummyThread("src-thread", "idle", {"graph_id": "agent"}, "test-user")
        source.metadata_json = {"graph_id": "agent"}

        with (
            patch("aegra_api.api.threads.copy_thread_checkpoints", AsyncMock()),
            patch("aegra_api.api.threads.db_manager"),
        ):
            response = self._client(_CopySession(source)).post("/threads/src-thread/copy")

        assert response.json()["metadata"]["graph_id"] == "agent"

    def test_missing_thread_is_404(self) -> None:
        with patch("aegra_api.api.threads.db_manager"):
            response = self._client(_CopySession(None)).post("/threads/nope/copy")

        assert response.status_code == 404


class _ThreadReadSession(DummySessionBase):
    def __init__(self, thread: Any) -> None:
        super().__init__()
        self.thread = thread

    async def scalar(self, stmt: Any = None) -> Any:
        return self.thread


class TestThreadInclude:
    def _client(self, thread: Any) -> TestClient:
        app = create_test_app()
        from aegra_api.core.orm import get_session as core_get_session

        app.dependency_overrides[core_get_session] = override_get_session_dep(lambda: _ThreadReadSession(thread))
        return make_client(app)

    def _thread(self) -> Any:
        thread = DummyThread("t-1", "idle", {"graph_id": "agent"}, "test-user")
        thread.metadata_json = {"graph_id": "agent"}
        return thread

    def test_config_and_state_timestamp_are_always_present(self) -> None:
        body = self._client(self._thread()).get("/threads/t-1").json()

        assert body["config"]["configurable"]["thread_id"] == "t-1"
        assert body["state_updated_at"] is not None

    def test_include_values_populates_state(self) -> None:
        with patch("aegra_api.api.threads._thread_state", AsyncMock(return_value=({"a": 1}, {"task": ["ask"]}))):
            body = self._client(self._thread()).get("/threads/t-1", params={"include": "values"}).json()

        assert body["values"] == {"a": 1}
        assert body["interrupts"] == {"task": ["ask"]}

    def test_include_ttl_populates_ttl(self) -> None:
        from aegra_api.models.threads import ThreadTTLInfo

        with patch(
            "aegra_api.api.threads._thread_ttl_info",
            AsyncMock(return_value=ThreadTTLInfo(strategy="delete", ttl=30)),
        ):
            body = self._client(self._thread()).get("/threads/t-1", params={"include": "ttl"}).json()

        assert body["ttl"]["strategy"] == "delete"

    def test_state_is_not_read_unless_requested(self) -> None:
        with patch("aegra_api.api.threads._thread_state", AsyncMock()) as reader:
            self._client(self._thread()).get("/threads/t-1")

        reader.assert_not_awaited()

    def test_unknown_include_is_rejected(self) -> None:
        response = self._client(self._thread()).get("/threads/t-1", params={"include": "everything"})

        assert response.status_code == 422


class _SearchSession(DummySessionBase):
    def __init__(self, threads: list[Any]) -> None:
        super().__init__()
        self.threads = threads

    async def scalars(self, stmt: Any = None) -> Any:
        return DummyScalarResult(self.threads)


class TestThreadSearchProjection:
    def _client(self, threads: list[Any]) -> TestClient:
        app = create_test_app()
        from aegra_api.core.orm import get_session as core_get_session

        app.dependency_overrides[core_get_session] = override_get_session_dep(lambda: _SearchSession(threads))
        return make_client(app)

    def _threads(self) -> list[Any]:
        thread = DummyThread("t-1", "idle", {}, "test-user")
        thread.metadata_json = {}
        return [thread]

    def test_select_returns_only_the_named_fields(self) -> None:
        response = self._client(self._threads()).post("/threads/search", json={"select": ["thread_id", "status"]})

        assert response.status_code == 200
        assert set(response.json()[0]) == {"thread_id", "status"}

    def test_without_select_the_full_thread_is_returned(self) -> None:
        body = self._client(self._threads()).post("/threads/search", json={}).json()

        assert {"thread_id", "status", "metadata", "created_at"} <= set(body[0])

    @pytest.mark.parametrize("payload", [{"values": {"a": 1}}, {"extract": {"alias": "values.a"}}])
    def test_state_filters_are_rejected(self, payload: dict[str, Any]) -> None:
        response = self._client(self._threads()).post("/threads/search", json=payload)

        assert response.status_code == 422


class TestRunBatch:
    def test_empty_batch_is_rejected(self, client: TestClient) -> None:
        assert client.post("/runs/batch", json=[]).status_code == 422

    def test_creates_one_run_per_payload(self, client: TestClient) -> None:
        created = [_run_model("r-1"), _run_model("r-2")]
        with patch(
            "aegra_api.api.stateless_runs.stateless_create_run",
            AsyncMock(side_effect=created),
        ) as creator:
            response = client.post(
                "/runs/batch",
                json=[
                    {"assistant_id": "agent", "input": {"messages": []}},
                    {"assistant_id": "agent", "input": {"messages": []}},
                ],
            )

        assert response.status_code == 200
        assert creator.await_count == 2

    def test_invalid_member_fails_the_batch(self, client: TestClient) -> None:
        response = client.post("/runs/batch", json=[{"input": {"messages": []}}])

        assert response.status_code == 422
