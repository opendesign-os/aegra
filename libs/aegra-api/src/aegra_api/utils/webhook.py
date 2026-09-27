"""Webhook URL validation and redaction shared by run and cron models."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

WEBHOOK_MAX_LEN = 65536


def validate_webhook_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("webhook must include a host")
    return value


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
