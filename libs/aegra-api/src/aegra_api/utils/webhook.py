"""Webhook URL validation shared by run and cron request models."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

WEBHOOK_MAX_LEN = 65536


def validate_webhook_url(value: str | None) -> str | None:
    """Reject malformed or non-http(s) webhook URLs at the API boundary.

    Records store ``webhook`` verbatim, so this is the SSRF entry point for
    every caller that later delivers to it.
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("webhook must include a host")
    return value


def mask_webhook_credentials(url: str) -> str:
    """Strip userinfo from a webhook URL before surfacing it back to clients."""
    parsed = urlparse(url)
    if "@" not in parsed.netloc:
        return url
    host = parsed.netloc.rsplit("@", 1)[1]
    return urlunparse(parsed._replace(netloc=host))
