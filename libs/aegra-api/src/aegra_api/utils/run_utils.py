import copy
from typing import Any

import structlog
from langgraph.types import Command, Send

logger = structlog.getLogger(__name__)

# A body-supplied thread_id overrides the route-verified one (the checkpointer
# keys on thread_id alone), so the server pins these instead of trusting them.
SERVER_PINNED_CONFIG_KEYS: frozenset[str] = frozenset({"thread_id", "run_id"})


def strip_pinned_config_keys(client_config: dict[str, Any]) -> dict[str, Any]:
    """Drop server-authoritative identity keys from a client-supplied config dict."""
    return {k: v for k, v in client_config.items() if k not in SERVER_PINNED_CONFIG_KEYS}


def map_command_to_langgraph(cmd: dict[str, Any]) -> Command:
    """Convert an API command dict to a LangGraph Command object."""
    goto = cmd.get("goto")
    if goto is not None and not isinstance(goto, list):
        goto = [goto]

    cmd_update = cmd.get("update")
    if isinstance(cmd_update, (tuple, list)) and all(
        isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str) for t in cmd_update
    ):
        cmd_update = [tuple(t) for t in cmd_update]

    return Command(
        update=cmd_update,
        goto=([it if isinstance(it, str) else Send(it["node"], it.get("input")) for it in goto] if goto else None),
        resume=cmd.get("resume"),
    )


def _should_skip_event(raw_event: Any) -> bool:
    """Check if an event should be skipped based on langsmith:nostream tag"""
    try:
        # Check if the event has metadata with tags containing 'langsmith:nostream'
        if isinstance(raw_event, tuple) and len(raw_event) >= 2:
            # For tuple events, check the third element (metadata tuple)
            metadata_tuple = raw_event[len(raw_event) - 1]
            if isinstance(metadata_tuple, tuple) and len(metadata_tuple) >= 2:
                # Get the second item in the metadata tuple
                metadata = metadata_tuple[1]
                if isinstance(metadata, dict) and "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list) and "langsmith:nostream" in tags:
                        return True
        return False
    except Exception:
        # If we can't parse the event structure, don't skip it
        return False


def _merge_jsonb(*objects: dict) -> dict:
    """Mimics PostgreSQL's JSONB merge behavior"""
    result = {}
    for obj in objects:
        if obj is not None:
            result.update(copy.deepcopy(obj))
    return result


def _configurable(config: dict[str, Any] | None) -> dict[str, Any] | None:
    value = (config or {}).get("configurable")
    return value if isinstance(value, dict) else None


def merge_run_config(assistant_config: dict[str, Any] | None, run_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a run's config over the assistant's.

    The top level follows PostgreSQL's JSONB ``||``: a key the run sets replaces
    the assistant's. ``configurable`` is merged one level deeper instead, the way
    LangChain's own ``merge_configs`` does it — replacing it wholesale would mean
    a run that pins one configurable key silently drops the model, and everything
    else the assistant had set alongside it.
    """
    merged = _merge_jsonb(assistant_config, run_config)
    configurable = _merge_jsonb(_configurable(assistant_config), _configurable(run_config))
    if configurable:
        merged["configurable"] = configurable
    return merged


async def _filter_context_by_schema(context: dict[str, Any], context_schema: dict | None) -> dict[str, Any]:
    """Filter context parameters based on the context schema."""
    if not context_schema or not context:
        return context

    # Extract valid properties from the schema
    properties = context_schema.get("properties", {})
    if not properties:
        return context

    filtered_context = {key: value for key, value in context.items() if key in properties}
    dropped = sorted(set(context) - set(filtered_context))
    if dropped:
        # Loud rather than debug: these keys never reach the graph, and the run
        # carries on as though the assistant had never set them.
        await logger.awarning(
            "Dropping context keys the graph does not declare",
            dropped_keys=dropped,
            declared_keys=sorted(properties),
        )
    return filtered_context


async def resolve_graph_context(graph: Any, context: dict[str, Any] | None) -> dict[str, Any]:
    """Trim a run's context to what the graph declares, for every streaming path.

    LangGraph builds the context object from the graph's own schema and raises a
    TypeError on a key it has never heard of, so an assistant carrying more than
    the graph declares has to be trimmed before the run starts. A graph that
    cannot describe its context — a JS graph, or one with no schema — takes the
    context as it stands.
    """
    if not context:
        return context or {}
    try:
        context_schema = graph.get_context_jsonschema()
    except Exception as exc:  # noqa: BLE001 — a graph that cannot describe itself still runs
        await logger.adebug(f"Failed to get context schema for filtering: {exc}", exc_info=exc)
        return context
    return await _filter_context_by_schema(context, context_schema)
