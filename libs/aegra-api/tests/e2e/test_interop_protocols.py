"""E2E coverage for the MCP and A2A interop endpoints.

Both speak JSON-RPC against a real server: MCP over the stateless Streamable
HTTP transport, A2A over plain POST with an SSE variant for `message/stream`.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings

pytestmark = pytest.mark.e2e

MCP_ACCEPT = "application/json, text/event-stream"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=180.0) as http_client:
        yield http_client


async def _mcp(client: httpx.AsyncClient, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = await client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}},
        headers={"accept": MCP_ACCEPT},
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestMcpTransport:
    async def test_initialize_reports_tool_capability(self, client: httpx.AsyncClient) -> None:
        body = await _mcp(
            client,
            "initialize",
            {
                "clientInfo": {"name": "e2e", "version": "1.0"},
                "protocolVersion": "2024-11-05",
                "capabilities": {},
            },
        )

        assert body["jsonrpc"] == "2.0"
        assert "tools" in body["result"]["capabilities"]
        assert body["result"]["serverInfo"]["name"]

    async def test_tools_list_exposes_the_deployments_graphs(self, client: httpx.AsyncClient) -> None:
        body = await _mcp(client, "tools/list")

        names = {tool["name"] for tool in body["result"]["tools"]}
        assert "agent" in names

    async def test_every_tool_declares_an_input_schema(self, client: httpx.AsyncClient) -> None:
        body = await _mcp(client, "tools/list")

        for tool in body["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object"

    async def test_accept_header_must_offer_both_media_types(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
            headers={"accept": "application/json"},
        )

        assert response.status_code == 400

    async def test_get_is_not_a_streaming_entry_point(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/mcp/")).status_code == 405

    async def test_delete_finds_no_session_to_terminate(self, client: httpx.AsyncClient) -> None:
        """The transport is stateless, so a session never exists to be deleted."""
        assert (await client.delete("/mcp/")).status_code == 404

    async def test_unknown_tool_is_an_error_not_a_crash(self, client: httpx.AsyncClient) -> None:
        body = await _mcp(client, "tools/call", {"name": "no-such-graph", "arguments": {}})

        assert body["result"]["isError"] is True


async def _a2a(client: httpx.AsyncClient, method: str, params: dict[str, Any], request_id: str = "1") -> dict[str, Any]:
    response = await client.post(
        "/a2a/agent", json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _message(**overrides: Any) -> dict[str, Any]:
    return {
        "message": {
            "role": "user",
            "parts": [{"kind": "data", "data": {"messages": []}}],
            "messageId": str(uuid.uuid4()),
            **overrides,
        }
    }


class TestA2AMessageSend:
    async def test_returns_a_task_in_the_spec_shape(self, client: httpx.AsyncClient) -> None:
        body = await _a2a(client, "message/send", _message())

        task = body["result"]
        assert task["kind"] == "task"
        assert task["contextId"]
        assert task["status"]["state"] in {"completed", "failed", "input-required"}

    async def test_context_id_is_the_thread(self, client: httpx.AsyncClient) -> None:
        first = (await _a2a(client, "message/send", _message()))["result"]

        second = (await _a2a(client, "message/send", _message(contextId=first["contextId"])))["result"]

        assert second["contextId"] == first["contextId"]
        assert second["id"] != first["id"]

    async def test_the_context_is_a_real_thread(self, client: httpx.AsyncClient) -> None:
        task = (await _a2a(client, "message/send", _message()))["result"]

        thread = await client.get(f"/threads/{task['contextId']}")

        assert thread.status_code == 200

    async def test_missing_message_is_invalid_params(self, client: httpx.AsyncClient) -> None:
        body = await _a2a(client, "message/send", {})

        assert body["error"]["code"] == -32602

    async def test_file_parts_are_declined_not_dropped(self, client: httpx.AsyncClient) -> None:
        """The base A2A spec allows file parts; this deployment cannot carry them."""
        body = await _a2a(
            client,
            "message/send",
            _message(parts=[{"kind": "file", "file": {"uri": "https://example/x.png"}}]),
        )

        assert body["error"]["code"] == -32005


class TestA2ATasks:
    async def test_get_returns_the_task(self, client: httpx.AsyncClient) -> None:
        created = (await _a2a(client, "message/send", _message()))["result"]

        body = await _a2a(client, "tasks/get", {"id": created["id"], "contextId": created["contextId"]})

        assert body["result"]["id"] == created["id"]

    async def test_unknown_task_is_reported(self, client: httpx.AsyncClient) -> None:
        body = await _a2a(client, "tasks/get", {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4())})

        assert body["error"]["code"] == -32001

    async def test_cancel_is_declined_rather_than_ignored(self, client: httpx.AsyncClient) -> None:
        body = await _a2a(client, "tasks/cancel", {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4())})

        assert body["error"]["code"] == -32002


class TestA2AProtocolErrors:
    async def test_unknown_method(self, client: httpx.AsyncClient) -> None:
        body = await _a2a(client, "no/such/method", {})

        assert body["error"]["code"] == -32601

    async def test_malformed_json_keeps_a_null_id(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/a2a/agent", content=b"{not json", headers={"content-type": "application/json"})

        body = response.json()
        assert body["error"]["code"] == -32700
        assert body["id"] is None

    async def test_wrong_jsonrpc_version_is_an_invalid_request(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/a2a/agent", json={"jsonrpc": "1.0", "id": "1", "method": "tasks/get", "params": {}}
        )

        assert response.json()["error"]["code"] == -32600


class TestA2AStreaming:
    async def test_stream_emits_task_state_transitions(self, client: httpx.AsyncClient) -> None:
        states: list[str] = []
        async with client.stream(
            "POST",
            "/a2a/agent",
            json={"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": _message()},
            headers={"accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                envelope = json.loads(line[5:])
                result = envelope.get("result")
                if result:
                    states.append(result["status"]["state"])

        assert states, "the stream produced no envelopes"
        assert states[-1] in {"completed", "failed", "canceled", "rejected"}
        assert states == sorted(set(states), key=states.index), "states must not repeat back to back"
