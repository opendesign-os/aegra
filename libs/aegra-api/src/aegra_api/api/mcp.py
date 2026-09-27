"""Model Context Protocol endpoint (stateless Streamable HTTP).

``POST /mcp/`` carries JSON-RPC 2.0 traffic; the session manager runs stateless,
so no session survives a request. ``GET`` is not a streaming entry point here and
``DELETE`` has no session to terminate.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.models import User
from aegra_api.services.mcp_server import mcp_transport

logger = structlog.getLogger(__name__)

router = APIRouter(tags=["MCP"], dependencies=auth_dependency)


class _TransportResponse(Response):
    """Hands the live ASGI channels to the MCP session manager.

    A Starlette response is itself an ASGI app, which is the only place a route
    handler can reach ``send``. FastAPI has already drained ``receive`` by the
    time we get here, so the body is replayed rather than re-read.
    """

    def __init__(self, user: User, body: bytes) -> None:
        super().__init__()
        self.user = user
        self.body_bytes = body

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": self.body_bytes, "more_body": False}
            return await receive()

        async with mcp_transport.bound_to(self.user):
            await mcp_transport.handle(scope, replay, send)


@router.post("/mcp/")
async def mcp_post(request: Request, user: User = Depends(get_current_user)) -> Response:
    """Send a JSON-RPC 2.0 message to the MCP server."""
    accept = request.headers.get("accept", "")
    if "application/json" not in accept or "text/event-stream" not in accept:
        raise HTTPException(
            400,
            "Accept header must include both 'application/json' and 'text/event-stream'.",
        )
    return _TransportResponse(user, await request.body())


@router.get("/mcp/")
async def mcp_get() -> Response:
    """Server-initiated streaming is not offered; the transport is stateless."""
    raise HTTPException(405, "GET is not supported; the MCP transport is stateless.")


@router.delete("/mcp/")
async def mcp_delete() -> Response:
    """No session is ever persisted, so there is none to terminate."""
    raise HTTPException(404, "Session not found; the MCP transport is stateless.")
