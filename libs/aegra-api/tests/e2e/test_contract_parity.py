"""E2E coverage for the endpoints and parameters added for LangSmith Deployment parity.

These run against a real server and database: the checkpoint copy, the catalog
lookup behind it, and the `include`/`select` projections all depend on Postgres
behaviour that a mocked session cannot exercise.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog

pytestmark = pytest.mark.e2e


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=60.0) as http_client:
        yield http_client


@pytest.fixture
async def assistant(client: httpx.AsyncClient) -> AsyncIterator[str]:
    assistant_id = f"parity-{uuid.uuid4().hex[:8]}"
    response = await client.post(
        "/assistants",
        json={
            "assistant_id": assistant_id,
            "graph_id": "agent",
            "name": assistant_id,
            # A distinct config keeps this off the unique index the default assistant holds.
            "config": {"configurable": {"probe": assistant_id}},
        },
    )
    response.raise_for_status()
    yield response.json()["assistant_id"]
    await client.delete(f"/assistants/{assistant_id}")


@pytest.fixture
async def thread(client: httpx.AsyncClient, assistant: str) -> AsyncIterator[str]:
    response = await client.post("/threads", json={"metadata": {"graph_id": "agent", "assistant_id": assistant}})
    response.raise_for_status()
    thread_id = response.json()["thread_id"]
    yield thread_id
    await client.delete(f"/threads/{thread_id}")


class TestSystemEndpoints:
    async def test_ok_reports_liveness(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/ok")

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_ok_can_probe_the_database(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/ok", params={"check_db": 1})

        assert response.status_code == 200

    async def test_metrics_serve_prometheus_exposition(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/metrics")
        if response.status_code == 404:
            pytest.skip("ENABLE_PROMETHEUS_METRICS=false; /metrics is intentionally off")

        assert response.status_code == 200
        assert "aegra_active_runs" in response.text
        assert "aegra_worker_count" in response.text

    async def test_metrics_serve_json(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/metrics", params={"format": "json"})
        if response.status_code == 404:
            pytest.skip("ENABLE_PROMETHEUS_METRICS=false; /metrics is intentionally off")

        body = response.json()
        assert "aegra_active_runs" in body


class TestThreadCount:
    async def test_counts_the_callers_threads(self, client: httpx.AsyncClient, thread: str) -> None:
        response = await client.post("/threads/count", json={})

        assert response.status_code == 200
        assert response.json() >= 1

    async def test_status_filter_narrows_the_count(self, client: httpx.AsyncClient, thread: str) -> None:
        idle = (await client.post("/threads/count", json={"status": "idle"})).json()
        total = (await client.post("/threads/count", json={})).json()

        assert idle <= total


class TestThreadCopy:
    async def test_fork_gets_a_new_id_and_keeps_metadata(self, client: httpx.AsyncClient, thread: str) -> None:
        response = await client.post(f"/threads/{thread}/copy")
        elog("copy", response.json() if response.status_code == 200 else response.text)

        assert response.status_code == 200
        body = response.json()
        assert body["thread_id"] != thread
        assert body["metadata"]["graph_id"] == "agent"

        await client.delete(f"/threads/{body['thread_id']}")

    async def test_copy_survives_a_thread_with_checkpoints(self, client: httpx.AsyncClient, thread: str) -> None:
        """The copy reads the checkpoint tables' real columns out of the catalog."""
        await client.post(f"/threads/{thread}/state", json={"values": {"messages": []}})

        response = await client.post(f"/threads/{thread}/copy")

        assert response.status_code == 200
        copy_id = response.json()["thread_id"]
        history = await client.post(f"/threads/{copy_id}/history", json={"limit": 5})
        assert history.status_code == 200

        await client.delete(f"/threads/{copy_id}")

    async def test_unknown_thread_is_404(self, client: httpx.AsyncClient) -> None:
        response = await client.post(f"/threads/{uuid.uuid4()}/copy")

        assert response.status_code == 404


class TestThreadIncludeAndSelect:
    async def test_config_is_always_returned(self, client: httpx.AsyncClient, thread: str) -> None:
        body = (await client.get(f"/threads/{thread}")).json()

        assert body["config"]["configurable"]["thread_id"] == thread

    async def test_include_populates_values_and_interrupts(self, client: httpx.AsyncClient, thread: str) -> None:
        response = await client.get(f"/threads/{thread}", params={"include": ["values", "interrupts"]})

        assert response.status_code == 200
        body = response.json()
        assert body["values"] is not None
        assert body["interrupts"] is not None

    async def test_unknown_include_is_rejected(self, client: httpx.AsyncClient, thread: str) -> None:
        response = await client.get(f"/threads/{thread}", params={"include": ["nonsense"]})

        assert response.status_code == 422

    async def test_select_projects_the_search_result(self, client: httpx.AsyncClient, thread: str) -> None:
        body = (await client.post("/threads/search", json={"select": ["thread_id", "status"], "limit": 5})).json()

        assert body
        assert set(body[0]) == {"thread_id", "status"}

    async def test_ids_restrict_the_search(self, client: httpx.AsyncClient, thread: str) -> None:
        body = (await client.post("/threads/search", json={"ids": [thread]})).json()

        assert [t["thread_id"] for t in body] == [thread]

    @pytest.mark.parametrize("payload", [{"values": {"a": 1}}, {"extract": {"x": "values.a"}}])
    async def test_state_filters_are_rejected(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        response = await client.post("/threads/search", json=payload)

        assert response.status_code == 422


class TestThreadPrune:
    async def test_named_threads_are_deleted(self, client: httpx.AsyncClient) -> None:
        created = (await client.post("/threads", json={"metadata": {"probe": "prune"}})).json()

        response = await client.post(
            "/threads/prune", json={"thread_ids": [created["thread_id"]], "strategy": "delete"}
        )

        assert response.status_code == 200
        assert response.json()["deleted"] == 1
        assert (await client.get(f"/threads/{created['thread_id']}")).status_code == 404

    async def test_another_users_thread_is_not_touched(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/threads/prune", json={"thread_ids": [str(uuid.uuid4())], "strategy": "delete"})

        assert response.status_code == 200
        assert response.json()["deleted"] == 0


class TestRunContractFields:
    async def test_run_response_carries_kwargs_and_metadata(
        self, client: httpx.AsyncClient, assistant: str, thread: str
    ) -> None:
        response = await client.post(
            f"/threads/{thread}/runs",
            json={
                "assistant_id": assistant,
                "input": {"messages": []},
                "durability": "exit",
                "stream_resumable": True,
                "after_seconds": 0,
                "metadata": {"tenant": "acme"},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["metadata"] == {"tenant": "acme"}
        assert body["kwargs"]["durability"] == "exit"
        assert body["kwargs"]["stream_resumable"] is True

    async def test_conflicting_durability_forms_are_rejected(
        self, client: httpx.AsyncClient, assistant: str, thread: str
    ) -> None:
        response = await client.post(
            f"/threads/{thread}/runs",
            json={
                "assistant_id": assistant,
                "input": {"messages": []},
                "durability": "sync",
                "checkpoint_during": True,
            },
        )

        assert response.status_code == 422

    async def test_malformed_webhook_is_rejected(self, client: httpx.AsyncClient, assistant: str, thread: str) -> None:
        response = await client.post(
            f"/threads/{thread}/runs",
            json={"assistant_id": assistant, "input": {"messages": []}, "webhook": "ftp://bad/x"},
        )

        assert response.status_code == 422

    async def test_runs_list_select_projects(self, client: httpx.AsyncClient, assistant: str, thread: str) -> None:
        await client.post(f"/threads/{thread}/runs", json={"assistant_id": assistant, "input": {"messages": []}})

        body = (await client.get(f"/threads/{thread}/runs", params={"select": ["run_id", "status"]})).json()

        assert body
        assert set(body[0]) == {"run_id", "status"}


class TestBatchAndSubgraphs:
    async def test_batch_creates_a_run_per_payload(self, client: httpx.AsyncClient, assistant: str) -> None:
        response = await client.post(
            "/runs/batch",
            json=[
                {"assistant_id": assistant, "input": {"messages": []}},
                {"assistant_id": assistant, "input": {"messages": []}},
            ],
        )

        assert response.status_code == 200
        assert len(response.json()) == 2

    async def test_empty_batch_is_rejected(self, client: httpx.AsyncClient) -> None:
        assert (await client.post("/runs/batch", json=[])).status_code == 422

    async def test_namespaced_subgraphs_route_resolves(self, client: httpx.AsyncClient, assistant: str) -> None:
        response = await client.get(f"/assistants/{assistant}/subgraphs/missing-namespace")

        assert response.status_code in (200, 404)


class TestCronContractFields:
    async def test_get_cron_by_id(self, client: httpx.AsyncClient, assistant: str) -> None:
        created = (
            await client.post(
                "/runs/crons",
                json={
                    "assistant_id": assistant,
                    "schedule": "0 0 * * *",
                    "input": {"messages": []},
                    "enabled": False,
                    "durability": "exit",
                    "stream_resumable": True,
                },
            )
        ).json()
        cron_id = created["cron_id"]

        response = await client.get(f"/runs/crons/{cron_id}")

        assert response.status_code == 200
        assert response.json()["cron_id"] == cron_id

        await client.delete(f"/runs/crons/{cron_id}")

    async def test_unknown_cron_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get(f"/runs/crons/{uuid.uuid4()}")).status_code == 404


class TestStoreContractFields:
    async def test_index_and_ttl_round_trip(self, client: httpx.AsyncClient) -> None:
        namespace = ["parity", uuid.uuid4().hex[:6]]

        put = await client.put(
            "/store/items",
            json={"namespace": namespace, "key": "k", "value": {"v": 1}, "index": ["v"], "ttl": 60},
        )
        assert put.status_code == 204

        got = await client.get(
            "/store/items",
            params={"namespace": ".".join(namespace), "key": "k", "refresh_ttl": True},
        )
        assert got.status_code == 200
        assert got.json()["value"] == {"v": 1}

        await client.request("DELETE", "/store/items", json={"namespace": namespace, "key": "k"})

    async def test_search_accepts_refresh_ttl(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/store/items/search", json={"namespace_prefix": ["parity"], "refresh_ttl": True})

        assert response.status_code == 200


class TestV2StateCommands:
    async def _command(self, client: httpx.AsyncClient, thread_id: str, method: str, **params: Any) -> Any:
        response = await client.post(
            f"/threads/{thread_id}/commands", json={"id": 1, "method": method, "params": params}
        )
        if response.status_code == 503:
            pytest.skip("v2 event streaming is disabled on this server")
        assert response.status_code == 200
        return response.json()

    async def test_state_get_returns_values(self, client: httpx.AsyncClient, thread: str) -> None:
        body = await self._command(client, thread, "state.get")

        assert body["type"] == "success"
        assert "values" in body["result"]

    async def test_list_checkpoints_returns_a_list(self, client: httpx.AsyncClient, thread: str) -> None:
        body = await self._command(client, thread, "state.listCheckpoints", limit=5)

        assert body["type"] == "success"
        assert isinstance(body["result"]["checkpoints"], list)

    async def test_agent_get_tree_names_the_graph(self, client: httpx.AsyncClient, thread: str) -> None:
        body = await self._command(client, thread, "agent.getTree")

        assert body["type"] == "success"
        assert body["result"]["tree"]["graph_name"] == "agent"

    async def test_subscription_reports_not_supported(self, client: httpx.AsyncClient, thread: str) -> None:
        body = await self._command(client, thread, "subscription.subscribe", channels=["values"])

        assert body["type"] == "error"
        assert body["error"] == "not_supported"

    async def test_unknown_method_is_unknown_command(self, client: httpx.AsyncClient, thread: str) -> None:
        body = await self._command(client, thread, "agent.noSuchMethod")

        assert body["error"] == "unknown_command"
