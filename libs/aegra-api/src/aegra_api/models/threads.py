"""Thread-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aegra_api.models.entity_ids import ENTITY_ID_PATTERN, MAX_ENTITY_ID_LENGTH
from aegra_api.models.search_limit import (
    resolve_search_limit,
    search_limit_json_schema_extra,
)
from aegra_api.utils.status_compat import validate_thread_status

# Upper bound keeping now + timedelta(minutes=ttl) finite and timedelta-safe
# (timedelta.max is ~1.44e9 minutes); rejects inf/1e308 at validation time.
MAX_TTL_MINUTES = 1_000_000_000

MAX_THREAD_ID_LENGTH = MAX_ENTITY_ID_LENGTH


class ThreadTTLSpec(BaseModel):
    """Per-thread TTL override supplied on thread creation.

    The langgraph SDK sends the minutes value as "ttl" while issue #288 names
    it "default_ttl" — AliasChoices accepts both spellings.
    """

    model_config = ConfigDict(populate_by_name=True)

    default_ttl: float | None = Field(
        None,
        gt=0,
        le=MAX_TTL_MINUTES,
        validation_alias=AliasChoices("default_ttl", "ttl"),
        description='Thread TTL in minutes; also accepted under the key "ttl" (LangGraph SDK '
        "compatibility). Falls back to the server default when omitted",
    )
    strategy: Literal["delete", "keep_latest"] | None = Field(
        None,
        description="Expiry strategy: 'delete' removes the thread, 'keep_latest' prunes history",
    )


class ThreadCreate(BaseModel):
    """Request model for creating threads"""

    model_config = ConfigDict(populate_by_name=True)

    metadata: dict[str, Any] | None = Field(None, description="Thread metadata")
    initial_state: dict[str, Any] | None = Field(None, description="LangGraph initial state")
    thread_id: str | None = Field(
        None,
        validation_alias=AliasChoices("thread_id", "threadId"),
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
        pattern=ENTITY_ID_PATTERN,
        description=(
            "Optional client-provided thread ID for idempotent creation. "
            "Omit or null to let the server generate a UUID. "
            f"When set, 1–{MAX_THREAD_ID_LENGTH} characters and not blank "
            "(must fit PostgreSQL btree keys uncompressed)."
        ),
    )
    if_exists: str | None = Field(
        "raise",
        validation_alias=AliasChoices("if_exists", "ifExists"),
        description="Behavior when thread exists: 'raise' (default) or 'do_nothing'",
    )
    ttl: ThreadTTLSpec | None = Field(
        None,
        description="Per-thread TTL override; requires TTL to be configured server-side or default_ttl set",
    )
    supersteps: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Checkpoint history to seed the thread with. Each superstep is "
            '{"updates": [{"values": ..., "as_node": ...}]}. Requires graph_id or '
            "assistant_id in metadata."
        ),
    )

    @field_validator("supersteps")
    @classmethod
    def validate_supersteps(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        for step in v:
            updates = step.get("updates")
            if not isinstance(updates, list) or not updates:
                raise ValueError("each superstep needs a non-empty 'updates' list")
            for update in updates:
                if not isinstance(update, dict):
                    raise ValueError("each update must be an object")
                if update.get("command") is not None:
                    raise ValueError("supersteps 'command' is not supported; use 'values' with 'as_node'")
        return v


class ThreadUpdate(BaseModel):
    """Request model for updating threads"""

    metadata: dict[str, Any] | None = Field(None, description="Thread metadata to update")
    ttl: ThreadTTLSpec | None = Field(None, description="Replace the thread's TTL")


class ThreadPruneRequest(BaseModel):
    """Request body for POST /threads/prune"""

    thread_ids: list[str] | None = Field(
        None,
        description="Threads to prune. Ids the caller cannot see are skipped. Omit to sweep expired threads.",
    )
    strategy: Literal["delete", "keep_latest"] = Field(
        "delete",
        description="'delete' removes the threads entirely; 'keep_latest' keeps each thread and its latest state.",
    )


class ThreadPruneResponse(BaseModel):
    """Response model for POST /threads/prune"""

    pruned_count: int = Field(0, description="Threads deleted or pruned by this request")
    deleted: int = Field(0, description="Threads fully deleted (strategy 'delete')")
    pruned: int = Field(0, description="Threads whose history was pruned (strategy 'keep_latest')")


class ThreadTTLInfo(BaseModel):
    """TTL state of a thread."""

    strategy: str | None = None
    ttl: float | None = None
    expires_at: datetime | None = None


class Thread(BaseModel):
    """Thread entity model

    Status values: idle, busy, interrupted, error
    """

    model_config = ConfigDict(from_attributes=True)

    thread_id: str = Field(..., description="Unique identifier for the thread.")
    status: str = Field("idle", description="Current thread status: idle, busy, interrupted, or error.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attached to the thread.")
    user_id: str = Field(..., description="Identifier of the user who owns this thread.")
    created_at: datetime = Field(..., description="Timestamp when the thread was created.")
    updated_at: datetime = Field(..., description="Timestamp when the thread was last updated.")
    state_updated_at: datetime | None = Field(None, description="Timestamp of the last state write.")
    config: dict[str, Any] | None = Field(None, description="Config the thread's runs execute with.")
    values: dict[str, Any] | None = Field(None, description="Current state values of the thread.")
    interrupts: dict[str, Any] | None = Field(None, description="Pending interrupts keyed by task id.")
    ttl: ThreadTTLInfo | None = Field(None, description="TTL info; only present when requested via include=ttl.")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status conforms to API specification."""
        if not isinstance(v, str):
            raise ValueError(f"Status must be a string, got {type(v)}")
        return validate_thread_status(v)


class ThreadList(BaseModel):
    """Response model for listing threads"""

    threads: list[Thread]
    total: int


ThreadSelectField = Literal[
    "thread_id", "created_at", "updated_at", "state_updated_at", "metadata", "config", "status", "values", "interrupts"
]


class ThreadCountRequest(BaseModel):
    """Request model for counting threads."""

    metadata: dict[str, Any] | None = Field(None, description="Metadata filters")
    values: dict[str, Any] | None = Field(None, description="State values to filter on")
    status: str | None = Field(None, description="Thread status filter (idle, busy, interrupted, error)")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        return validate_thread_status(v) if v is not None else v


class ThreadSearchRequest(BaseModel):
    """Request model for thread search"""

    metadata: dict[str, Any] | None = Field(None, description="Metadata filters")
    status: str | None = Field(None, description="Thread status filter (idle, busy, interrupted, error)")
    # None default + validate_default so omitted and JSON null share one resolver.
    limit: int | None = Field(
        default=None,
        ge=1,
        validate_default=True,
        description="Maximum results",
        json_schema_extra=search_limit_json_schema_extra,
    )
    offset: int | None = Field(0, ge=0, description="Results offset")
    order_by: str | None = Field(
        "created_at DESC",
        deprecated=True,
        description="DEPRECATED: use sort_by + sort_order. Legacy single-field form, e.g. 'updated_at ASC'.",
    )
    sort_by: Literal["thread_id", "status", "created_at", "updated_at"] | None = Field(
        None,
        description="Field to sort by (SDK-compatible). Takes precedence over order_by.",
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        None,
        description="Sort direction (SDK-compatible). Defaults to 'desc' when sort_by is set.",
    )
    ids: list[str] | None = Field(None, description="Restrict results to these thread IDs")
    select: list[ThreadSelectField] | None = Field(
        None, exclude=True, description="Fields to return; all fields when omitted"
    )
    values: dict[str, Any] | None = Field(None, description="State values to filter on")
    extract: dict[str, str] | None = Field(None, description="Alias to JSONB path projections")

    @field_validator("values", "extract")
    @classmethod
    def reject_state_filters(cls, v: Any, info: ValidationInfo) -> Any:
        """Aegra keeps state in checkpoints, not on the thread row, so these cannot be served."""
        if v:
            raise ValueError(
                f"'{info.field_name}' is not supported: thread state lives in checkpoints, "
                "not on the thread row. Read state via /threads/{thread_id}/state."
            )
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls: type["ThreadSearchRequest"], v: int | None) -> int:
        return resolve_search_limit(v)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        """Validate status filter conforms to API specification."""
        if v is not None:
            return validate_thread_status(v)
        return v


class ThreadSearchResponse(BaseModel):
    """Response model for thread search"""

    threads: list[Thread]
    total: int
    limit: int
    offset: int


class ThreadCheckpoint(BaseModel):
    """Checkpoint identifier for thread history"""

    checkpoint_id: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = ""


class ThreadCheckpointPostRequest(BaseModel):
    """Request model for fetching thread checkpoint"""

    checkpoint: ThreadCheckpoint = Field(description="Checkpoint to fetch")
    subgraphs: bool | None = Field(False, description="Include subgraph states")


class ThreadState(BaseModel):
    """Thread state model for history endpoint"""

    values: dict[str, Any] = Field(description="Channel values (messages, etc.)")
    next: list[str] = Field(default_factory=list, description="Next nodes to execute")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="Tasks to execute")
    interrupts: list[dict[str, Any]] = Field(default_factory=list, description="Interrupt data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checkpoint metadata")
    created_at: datetime | None = Field(None, description="Timestamp of state creation")
    checkpoint: ThreadCheckpoint = Field(description="Current checkpoint")
    parent_checkpoint: ThreadCheckpoint | None = Field(None, description="Parent checkpoint")
    checkpoint_id: str | None = Field(None, description="Checkpoint ID (for backward compatibility)")
    parent_checkpoint_id: str | None = Field(None, description="Parent checkpoint ID (for backward compatibility)")


class ThreadStateUpdate(BaseModel):
    """Request model for updating thread state"""

    values: dict[str, Any] | list[dict[str, Any]] | None = Field(
        None, description="The values to update the state with"
    )
    checkpoint: dict[str, Any] | None = Field(None, description="The checkpoint to update the state of")
    checkpoint_id: str | None = Field(None, description="Optional checkpoint ID to update from")
    as_node: str | None = Field(None, description="Update the state as if this node had just executed")
    # Also support query-like parameters for GET-like behavior via POST
    subgraphs: bool | None = Field(False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(None, description="Checkpoint namespace")


class ThreadStateUpdateResponse(BaseModel):
    """Response model for thread state update"""

    checkpoint: dict[str, Any] = Field(description="The checkpoint that was created/updated")


class ThreadHistoryRequest(BaseModel):
    """Request model for thread history endpoint"""

    limit: int | None = Field(10, ge=1, le=1000, description="Number of states to return")
    before: dict[str, Any] | str | None = Field(
        None,
        description="Return states before this checkpoint (checkpoint ID string, raw checkpoint dict, or RunnableConfig with 'configurable' key)",
    )
    metadata: dict[str, Any] | None = Field(None, description="Filter by metadata")
    checkpoint: dict[str, Any] | None = Field(None, description="Checkpoint for subgraph filtering")
    subgraphs: bool | None = Field(False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(None, description="Checkpoint namespace")
