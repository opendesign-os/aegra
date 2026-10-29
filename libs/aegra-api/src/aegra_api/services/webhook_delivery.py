"""Deliver the terminal run payload to a caller-supplied webhook."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from functools import cache
from typing import Any
from uuid import uuid4

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from aegra_api.settings import settings
from aegra_api.utils.webhook import check_public_target, mask_webhook_credentials

logger = structlog.getLogger(__name__)

ATTEMPTS = 3
_TIMEOUT_SECS = 10.0

# Standard Webhooks (https://www.standardwebhooks.com): the signature covers
# "{id}.{timestamp}.{body}" under HMAC-SHA256, and keys carry a whsec_ prefix
# with the raw key base64-encoded behind it.
_SIGNATURE_VERSION = "v1"
_SECRET_PREFIX = "whsec_"


def _secret() -> bytes | None:
    raw = settings.webhook.AEGRA_WEBHOOK_SECRET
    if not raw:
        return None
    if raw.startswith(_SECRET_PREFIX):
        return base64.b64decode(raw[len(_SECRET_PREFIX) :])
    return raw.encode()


@cache
def _warn_unsigned() -> None:
    logger.warning(
        "Webhooks are going out unsigned; set AEGRA_WEBHOOK_SECRET so receivers can verify them",
    )


def _headers(message_id: str, body: bytes, secret: bytes | None) -> dict[str, str]:
    """Build the delivery headers, signed when a secret is configured.

    The id is the same on every attempt so a receiver can deduplicate retries on
    it, while each attempt is signed with the time it is sent so a late retry
    does not land outside the receiver's replay window.
    """
    headers = {"content-type": "application/json", "webhook-id": message_id}
    if secret is None:
        return headers

    timestamp = str(int(time.time()))
    signature = hmac.new(secret, f"{message_id}.{timestamp}.".encode() + body, hashlib.sha256).digest()
    headers["webhook-timestamp"] = timestamp
    headers["webhook-signature"] = f"{_SIGNATURE_VERSION},{base64.b64encode(signature).decode()}"
    return headers


async def _post_once(url: str, body: bytes, message_id: str, secret: bytes | None, masked: str) -> bool:
    try:
        # Redirects stay off: following one is the ordinary way around a host
        # check that only ever saw the original URL.
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECS, follow_redirects=False) as client:
            response = await client.post(url, content=body, headers=_headers(message_id, body, secret))
    except httpx.HTTPError as exc:
        logger.warning("Webhook delivery failed", webhook=masked, error=str(exc))
        return False

    if response.status_code < 400:
        logger.info("Webhook delivered", webhook=masked, status=response.status_code)
        return True
    logger.warning("Webhook rejected", webhook=masked, status=response.status_code)
    return False


async def deliver(url: str, payload: dict[str, Any]) -> bool:
    masked = mask_webhook_credentials(url)
    try:
        await check_public_target(url)
    except ValueError as exc:
        logger.error("Webhook refused", webhook=masked, error=str(exc))
        return False

    try:
        body = json.dumps(payload, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        logger.error("Webhook payload could not be serialized", webhook=masked, error=str(exc))
        return False

    secret = _secret()
    if secret is None:
        _warn_unsigned()
    message_id = str(uuid4())

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(ATTEMPTS),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_result(lambda delivered: delivered is False),
            reraise=False,
        ):
            with attempt:
                delivered = await _post_once(url, body, message_id, secret, masked)
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(delivered)
    except RetryError:
        logger.error("Webhook exhausted retries", webhook=masked)
        return False
    return delivered


def schedule(url: str | None, payload: dict[str, Any]) -> None:
    """Fire-and-forget delivery so a slow endpoint never stalls run finalization."""
    if not url:
        return
    task = asyncio.create_task(deliver(url, payload))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


_pending: set[asyncio.Task[bool]] = set()
