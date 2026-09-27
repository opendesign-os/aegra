"""Optional Prometheus metrics via prometheus-fastapi-instrumentator.

Controlled by ``ENABLE_PROMETHEUS_METRICS`` env var (default: false).
When enabled, exposes a ``/metrics`` endpoint with standard HTTP and
Python runtime metrics in Prometheus exposition format.
"""

from typing import Any, Literal

import prometheus_client
import structlog
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from aegra_api.core.active_runs import active_runs
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

REAPER_RECOVERED_RUNS = prometheus_client.Counter(
    "aegra_reaper_recovered_runs_total",
    "Runs recovered by the lease reaper, by outcome: crashed_retried "
    "(expired lease, re-enqueued), crashed_exhausted (max retries exceeded, "
    "marked failed), stuck_pending (missing from the queue, re-enqueued). Counts only "
    "confirmed Redis pushes and DB updates; recovery that falls back to the "
    "workers' Postgres poll during a Redis outage is not counted.",
    labelnames=["outcome"],
)

# Pre-create label children so every outcome series renders as 0 on /metrics
# before the first recovery event (absent series break rate() alerts).
for _outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
    REAPER_RECOVERED_RUNS.labels(outcome=_outcome)

THREAD_TTL_SWEPT = prometheus_client.Counter(
    "aegra_thread_ttl_swept_threads_total",
    "Threads processed by the TTL sweep or POST /threads/prune, by outcome: "
    "deleted (strategy=delete, thread removed), pruned (strategy=keep_latest, "
    "history compacted), error (item failed and will be retried next tick).",
    labelnames=["outcome"],
)

for _outcome in ("deleted", "pruned", "error"):
    THREAD_TTL_SWEPT.labels(outcome=_outcome)

ACTIVE_RUNS = prometheus_client.Gauge(
    "aegra_active_runs",
    "Runs currently executing in this instance.",
)
ACTIVE_RUNS.set_function(lambda: len(active_runs))

WORKER_COUNT = prometheus_client.Gauge(
    "aegra_worker_count",
    "Workers configured on this instance.",
)
JOBS_PER_WORKER = prometheus_client.Gauge(
    "aegra_jobs_per_worker",
    "Concurrent jobs each worker accepts.",
)


def setup_prometheus_metrics(
    app: FastAPI,
    registry: prometheus_client.CollectorRegistry | None = None,
) -> None:
    """Conditionally attach Prometheus instrumentator to the app.

    No-op when ``ENABLE_PROMETHEUS_METRICS`` is false.

    Args:
        app: FastAPI application instance.
        registry: Optional Prometheus collector registry. When provided, metrics
            are collected into this registry instead of the global default.
            Primarily useful in tests to avoid cross-test pollution.

    Note:
        The ``/metrics`` endpoint is **not** protected by Aegra's authentication
        middleware. This is intentional — Prometheus scrapers typically do not
        support application-level auth. If the endpoint must be restricted, use
        network-level controls (firewall rules, internal load-balancer, etc.).
    """
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        return

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/ready", "/live", "/info", "/metrics", "/docs", "/redoc", "/openapi.json"],
        registry=registry,
    )
    instrumentator.instrument(app)
    WORKER_COUNT.set(settings.worker.WORKER_COUNT)
    JOBS_PER_WORKER.set(settings.worker.N_JOBS_PER_WORKER)

    collector_registry = registry or prometheus_client.REGISTRY

    @app.get("/metrics", include_in_schema=False)
    async def metrics(
        format: Literal["prometheus", "json"] = Query("prometheus"),
    ) -> Response:
        """Prometheus exposition, or the same families as JSON when asked."""
        if format == "json":
            return JSONResponse(_as_json(collector_registry))
        return Response(
            prometheus_client.generate_latest(collector_registry),
            media_type=prometheus_client.CONTENT_TYPE_LATEST,
        )

    logger.info("Prometheus metrics enabled at /metrics")


def _as_json(registry: prometheus_client.CollectorRegistry) -> dict[str, Any]:
    """Flatten the registry into {metric: value} / {metric: {labels: value}}."""
    out: dict[str, Any] = {}
    for family in registry.collect():
        for sample in family.samples:
            if sample.labels:
                key = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
                out.setdefault(sample.name, {})[key] = sample.value
            else:
                out[sample.name] = sample.value
    return out
