"""HTTP-level tests for contract parameters added to existing endpoints."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.api import assistants as assistants_module
from aegra_api.api import crons as crons_module
from aegra_api.api import store as store_module
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.models.auth import User

USER = User(identity="test-user", display_name="Test User")


def _app(router: Any, dependency: Any = None, provider: Any = None) -> FastAPI:
    app = FastAPI()
    app.dependency_overrides[require_auth] = lambda: USER
    app.dependency_overrides[get_current_user] = lambda: USER
    if dependency is not None:
        app.dependency_overrides[dependency] = provider
    app.include_router(router)
    return app


def _assistant(assistant_id: str = "a-1") -> Any:
    from datetime import UTC, datetime

    from aegra_api.models.assistants import Assistant

    now = datetime.now(UTC)
    return Assistant(
        assistant_id=assistant_id,
        name="one",
        graph_id="agent",
        user_id="test-user",
        version=1,
        created_at=now,
        updated_at=now,
    )


# --- assistants ------------------------------------------------------------


@pytest.fixture
def assistant_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def assistant_client(assistant_service: AsyncMock) -> TestClient:
    from aegra_api.services.assistant_service import get_assistant_service

    return TestClient(_app(assistants_module.router, get_assistant_service, lambda: assistant_service))


class TestSubgraphsByNamespace:
    def test_path_namespace_reaches_the_service(
        self, assistant_client: TestClient, assistant_service: AsyncMock
    ) -> None:
        assistant_service.get_assistant_subgraphs.return_value = {}

        response = assistant_client.get("/assistants/a-1/subgraphs/inner")

        assert response.status_code == 200
        assistant_service.get_assistant_subgraphs.assert_awaited_once_with("a-1", "inner", False)

    def test_recurse_is_forwarded(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.get_assistant_subgraphs.return_value = {}

        assistant_client.get("/assistants/a-1/subgraphs/inner", params={"recurse": True})

        assistant_service.get_assistant_subgraphs.assert_awaited_once_with("a-1", "inner", True)

    def test_query_form_still_works(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.get_assistant_subgraphs.return_value = {}

        assistant_client.get("/assistants/a-1/subgraphs", params={"namespace": "inner"})

        assistant_service.get_assistant_subgraphs.assert_awaited_once_with("a-1", "inner", False)


class TestDeleteThreads:
    def test_defaults_to_keeping_threads(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.delete_assistant.return_value = {"status": "deleted"}

        assistant_client.delete("/assistants/a-1")

        assistant_service.delete_assistant.assert_awaited_once_with("a-1", delete_threads=False)

    def test_opt_in_is_forwarded(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.delete_assistant.return_value = {"status": "deleted", "deleted_threads": 2}

        response = assistant_client.delete("/assistants/a-1", params={"delete_threads": True})

        assistant_service.delete_assistant.assert_awaited_once_with("a-1", delete_threads=True)
        assert response.json()["deleted_threads"] == 2


class TestSetLatestVersion:
    def test_accepts_the_query_form(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.set_assistant_latest.return_value = _assistant()

        response = assistant_client.post("/assistants/a-1/latest", params={"version": 3})

        assert response.status_code == 200
        assistant_service.set_assistant_latest.assert_awaited_once_with("a-1", 3)

    def test_accepts_the_body_form(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.set_assistant_latest.return_value = _assistant()

        assistant_client.post("/assistants/a-1/latest", json={"version": 4})

        assistant_service.set_assistant_latest.assert_awaited_once_with("a-1", 4)

    def test_query_wins_over_body(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.set_assistant_latest.return_value = _assistant()

        assistant_client.post("/assistants/a-1/latest", params={"version": 5}, json={"version": 9})

        assistant_service.set_assistant_latest.assert_awaited_once_with("a-1", 5)

    def test_missing_version_is_rejected(self, assistant_client: TestClient) -> None:
        assert assistant_client.post("/assistants/a-1/latest", json={}).status_code == 422


class TestAssistantSelect:
    def test_select_projects_the_response(self, assistant_client: TestClient, assistant_service: AsyncMock) -> None:
        assistant_service.search_assistants.return_value = [_assistant()]

        response = assistant_client.post("/assistants/search", json={"select": ["assistant_id", "name"]})

        assert set(response.json()[0]) == {"assistant_id", "name"}


# --- crons -----------------------------------------------------------------


@pytest.fixture
def cron_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def cron_client(cron_service: AsyncMock) -> TestClient:
    from aegra_api.services.cron_service import get_cron_service

    return TestClient(_app(crons_module.router, get_cron_service, lambda: cron_service))


def _cron_response(cron_id: str = "c-1") -> Any:
    from datetime import UTC, datetime

    from aegra_api.models.crons import CronResponse

    now = datetime.now(UTC)
    return CronResponse(
        cron_id=cron_id,
        assistant_id="agent",
        schedule="0 0 * * *",
        created_at=now,
        updated_at=now,
    )


class TestGetCron:
    def test_reads_one_cron(self, cron_client: TestClient, cron_service: AsyncMock) -> None:
        cron_service.get_cron.return_value = _cron_response()

        response = cron_client.get("/runs/crons/c-1")

        assert response.status_code == 200
        assert response.json()["cron_id"] == "c-1"
        cron_service.get_cron.assert_awaited_once_with("c-1", "test-user")


class TestCronSelect:
    def test_select_projects_the_response(self, cron_client: TestClient, cron_service: AsyncMock) -> None:
        cron_service.search_crons.return_value = [_cron_response()]

        response = cron_client.post("/runs/crons/search", json={"select": ["cron_id", "schedule"]})

        assert set(response.json()[0]) == {"cron_id", "schedule"}


class TestCronDurabilityFields:
    """The cron payload must carry what a fired run needs, or the value is dropped at firing."""

    def test_create_accepts_durability_and_resumable(self) -> None:
        from aegra_api.models.crons import CronCreate

        request = CronCreate.model_validate(
            {
                "assistant_id": "agent",
                "schedule": "0 0 * * *",
                "input": {"messages": []},
                "durability": "exit",
                "stream_resumable": True,
            }
        )

        assert request.durability == "exit"
        assert request.stream_resumable is True

    def test_payload_persists_them(self) -> None:
        from aegra_api.models.crons import CronCreate
        from aegra_api.services.cron_service import _build_payload

        request = CronCreate.model_validate(
            {
                "assistant_id": "agent",
                "schedule": "0 0 * * *",
                "input": {"messages": []},
                "durability": "sync",
                "stream_resumable": True,
                "webhook": "https://hooks.example/cb",
            }
        )
        payload = _build_payload(request)

        assert payload["durability"] == "sync"
        assert payload["stream_resumable"] is True
        assert payload["webhook"] == "https://hooks.example/cb"

    def test_fired_run_carries_them(self) -> None:
        from aegra_api.services.cron_scheduler import _build_run_create

        class _Cron:
            assistant_id = "agent"
            payload = {
                "input": {"messages": []},
                "durability": "exit",
                "stream_resumable": True,
                "webhook": "https://hooks.example/cb",
            }

        request = _build_run_create(_Cron())

        assert request.resolved_durability == "exit"
        assert request.stream_resumable is True
        assert request.webhook == "https://hooks.example/cb"


# --- store -----------------------------------------------------------------


class _Store:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    async def aput(self, **kwargs: Any) -> None:
        self.put_calls.append(kwargs)

    async def aget(self, namespace: Any, key: str, **kwargs: Any) -> Any:
        self.get_calls.append({"namespace": namespace, "key": key, **kwargs})

        class _Item:
            value = {"v": 1}

        return _Item()

    async def asearch(self, prefix: Any, **kwargs: Any) -> list[Any]:
        self.search_calls.append({"prefix": prefix, **kwargs})
        return []


@pytest.fixture
def store() -> _Store:
    return _Store()


@pytest.fixture
def store_client(store: _Store) -> TestClient:
    app = _app(store_module.router)
    with patch("aegra_api.api.store.db_manager") as manager:
        manager.get_store.return_value = store
        yield TestClient(app)


class TestStoreIndexAndTTL:
    def test_put_forwards_index_and_ttl(self, store_client: TestClient, store: _Store) -> None:
        response = store_client.put(
            "/store/items",
            json={"namespace": ["ns"], "key": "k", "value": {"v": 1}, "index": ["v"], "ttl": 60},
        )

        assert response.status_code == 204
        assert store.put_calls[0]["index"] == ["v"]
        assert store.put_calls[0]["ttl"] == 60

    def test_put_without_ttl_omits_the_kwarg(self, store_client: TestClient, store: _Store) -> None:
        """Stores that predate TTL must not receive an unexpected keyword."""
        store_client.put("/store/items", json={"namespace": ["ns"], "key": "k", "value": {"v": 1}})

        assert "ttl" not in store.put_calls[0]

    def test_get_forwards_refresh_ttl(self, store_client: TestClient, store: _Store) -> None:
        store_client.get("/store/items", params={"namespace": "ns", "key": "k", "refresh_ttl": True})

        assert store.get_calls[0]["refresh_ttl"] is True

    def test_get_without_refresh_ttl_omits_the_kwarg(self, store_client: TestClient, store: _Store) -> None:
        store_client.get("/store/items", params={"namespace": "ns", "key": "k"})

        assert "refresh_ttl" not in store.get_calls[0]

    def test_search_forwards_refresh_ttl(self, store_client: TestClient, store: _Store) -> None:
        store_client.post("/store/items/search", json={"namespace_prefix": ["ns"], "refresh_ttl": True})

        assert store.search_calls[0]["refresh_ttl"] is True
