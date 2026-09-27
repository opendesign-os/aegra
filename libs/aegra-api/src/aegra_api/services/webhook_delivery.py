"""Deliver the terminal run payload to a caller-supplied webhook."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from aegra_api.utils.webhook import mask_webhook_credentials

logger = structlog.getLogger(__name__)

_ATTEMPTS = 3
_BACKOFF_SECS = 2.0
_TIMEOUT_SECS = 10.0


async def deliver(url: str, payload: dict[str, Any]) -> bool:
    masked = mask_webhook_credentials(url)
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECS) as client:
                response = await client.post(url, json=payload)
            if response.status_code < 400:
                logger.info("Webhook delivered", webhook=masked, status=response.status_code)
                return True
            logger.warning("Webhook rejected", webhook=masked, status=response.status_code, attempt=attempt)
        except httpx.HTTPError as exc:
            logger.warning("Webhook delivery failed", webhook=masked, attempt=attempt, error=str(exc))
        if attempt < _ATTEMPTS:
            await asyncio.sleep(_BACKOFF_SECS * attempt)
    logger.error("Webhook exhausted retries", webhook=masked)
    return False


def schedule(url: str | None, payload: dict[str, Any]) -> None:
    """Fire-and-forget delivery so a slow endpoint never stalls run finalization."""
    if not url:
        return
    task = asyncio.create_task(deliver(url, payload))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


_pending: set[asyncio.Task[bool]] = set()
