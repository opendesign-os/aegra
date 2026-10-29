"""Webhook URL validation, SSRF guarding and redaction shared by run and cron models."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

import structlog

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

WEBHOOK_MAX_LEN = 65536

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _allowed_hosts() -> frozenset[str]:
    raw = settings.webhook.AEGRA_WEBHOOK_ALLOWED_HOSTS
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def _guarded(host: str) -> bool:
    """False when this deployment has opted the host out of the public-address rule."""
    if settings.webhook.AEGRA_WEBHOOK_ALLOW_PRIVATE_NETWORKS:
        return False
    return host.lower() not in _allowed_hosts()


def _reject_internal(host: str, addresses: list[str]) -> None:
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError(
                f"webhook host '{host}' points at the non-public address {address}; "
                "name it in AEGRA_WEBHOOK_ALLOWED_HOSTS or set "
                "AEGRA_WEBHOOK_ALLOW_PRIVATE_NETWORKS=true to allow it"
            )


def _known_address(host: str) -> str | None:
    """The address a host has without asking DNS, when that is knowable at all."""
    lowered = host.lower()
    # RFC 6761: localhost and everything under it always means loopback, so the
    # commonest internal target can be answered for at request time.
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return "127.0.0.1"
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return None
    return host


def validate_webhook_url(value: str | None) -> str | None:
    """Check a caller-supplied webhook URL at request time.

    Names are left alone here: resolving them would put DNS on the request path,
    and the answer can change before delivery anyway. ``check_public_target``
    resolves and checks again when the callback actually goes out.
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    host = parsed.hostname
    if not host:
        raise ValueError("webhook must include a host")

    known = _known_address(host)
    if known is not None and _guarded(host):
        _reject_internal(host, [known])
    return value


async def _resolve(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


async def check_public_target(url: str) -> None:
    """Raise when the webhook host resolves anywhere but publicly routable space.

    httpx resolves the name again when it connects, so a name that flips to an
    internal address in between still gets through. This closes the plain case —
    a URL aimed at localhost, a cloud metadata service or a neighbour on the
    private network — not a deliberate rebind.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not _guarded(host):
        return

    known = _known_address(host)
    if known is not None:
        _reject_internal(host, [known])
        return

    try:
        addresses = await _resolve(host, parsed.port or _DEFAULT_PORTS[parsed.scheme])
    except OSError as exc:
        # A name that will not resolve cannot be connected to either, so the
        # delivery attempt fails on its own and keeps its retries.
        logger.debug("Webhook host did not resolve", host=host, error=str(exc))
        return
    _reject_internal(host, addresses)


def mask_webhook_credentials(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.hostname:
        return url
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))
