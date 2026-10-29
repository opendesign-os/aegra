"""Expose the deployment's graphs as MCP tools over stateless streamable HTTP.

One tool per graph, each invoking a stateless run through the normal execution
path so the call shows up in run history, tracing and cancellation like any
other run.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import mcp.types as mcp_types
import structlog
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from aegra_api import __version__
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_waiters import read_run_result
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# The ASGI handler runs outside FastAPI's dependency graph, so the authenticated
# user is handed over on a contextvar set by the route.
current_user: ContextVar[User | None] = ContextVar("mcp_current_user", default=None)

_DEFAULT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "object", "description": "Graph input."}},
    "required": ["input"],
}


async def _tool_input_schema(graph_id: str, user: User | None) -> dict[str, Any]:
    try:
        async with get_langgraph_service().get_graph(graph_id, access_context="assistants.read", user=user) as graph:
            return graph.get_input_jsonschema() or _DEFAULT_INPUT_SCHEMA
    except Exception:  # noqa: BLE001 — a graph that cannot describe itself still gets a generic tool
        logger.debug("mcp: falling back to the generic input schema", graph_id=graph_id)
        return _DEFAULT_INPUT_SCHEMA


async def _list_tools(
    _context: ServerRequestContext[Any],
    _params: mcp_types.PaginatedRequestParams | None,
) -> mcp_types.ListToolsResult:
    user = current_user.get()
    return mcp_types.ListToolsResult(
        tools=[
            mcp_types.Tool(
                name=graph_id,
                description=f"Run the '{graph_id}' graph and return its final output.",
                inputSchema=await _tool_input_schema(graph_id, user),
            )
            for graph_id in get_langgraph_service().list_graphs()
        ]
    )


def _error_result(message: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=message)], isError=True)


async def _call_tool(
    _context: ServerRequestContext[Any],
    params: mcp_types.CallToolRequestParams,
) -> mcp_types.CallToolResult:
    user = current_user.get()
    if user is None:
        return _error_result("MCP tool calls require an authenticated caller.")
    if params.name not in get_langgraph_service().list_graphs():
        return _error_result(f"Unknown tool {params.name!r}.")

    output = await _run_graph(params.name, params.arguments or {}, user)
    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=str(output))])


def build_server() -> Server[Any]:
    return Server(
        settings.app.PROJECT_NAME,
        version=__version__,
        instructions="Each tool runs one graph of this deployment and returns its final output.",
        on_list_tools=_list_tools,
        on_call_tool=_call_tool,
    )


async def _run_graph(graph_id: str, arguments: dict[str, Any], user: User) -> Any:
    """Execute one stateless run and return its final output."""
    payload = arguments.get("input") if set(arguments) == {"input"} else arguments
    thread_id = str(uuid4())

    maker = _get_session_maker()
    async with maker() as session:
        run_id, _run, _job = await _prepare_run(
            session,
            thread_id,
            RunCreate(assistant_id=graph_id, input=payload or {}),
            user,
            initial_status="pending",
        )

    with contextlib.suppress(TimeoutError):
        await executor.wait_for_completion(run_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)

    return await read_run_result(run_id, thread_id, user.identity)


class McpTransport:
    """Owns the stateless session manager and its lifespan."""

    def __init__(self) -> None:
        self._manager: StreamableHTTPSessionManager | None = None
        self._exit: contextlib.AsyncExitStack | None = None

    async def start(self) -> None:
        # json_response=True: the documented contract answers application/json,
        # not an SSE stream.
        manager = StreamableHTTPSessionManager(app=build_server(), stateless=True, json_response=True)
        stack = contextlib.AsyncExitStack()
        await stack.enter_async_context(manager.run())
        self._manager, self._exit = manager, stack
        logger.info("MCP transport started at /mcp/")

    async def stop(self) -> None:
        if self._exit is not None:
            await self._exit.aclose()
        self._manager, self._exit = None, None

    async def handle(self, scope: Any, receive: Any, send: Any) -> None:
        if self._manager is None:
            raise RuntimeError("MCP transport is not running")
        await self._manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def bound_to(self, user: User) -> AsyncIterator[None]:
        token = current_user.set(user)
        try:
            yield
        finally:
            current_user.reset(token)


mcp_transport = McpTransport()
