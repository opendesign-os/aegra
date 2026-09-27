"""Checkpoint-lineage copy: column lists come from the catalog, thread_id is rebound."""

from typing import Any

import pytest
from psycopg import sql

from aegra_api.services.thread_copy import _CHECKPOINT_TABLES, copy_thread_checkpoints

_COLUMNS = {
    "checkpoints": ["thread_id", "checkpoint_ns", "checkpoint_id", "parent_checkpoint_id", "checkpoint"],
    "checkpoint_blobs": ["thread_id", "checkpoint_ns", "channel", "version", "blob"],
    "checkpoint_writes": ["thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx", "blob"],
}


class _Cursor:
    """Mimics the LangGraph pool's dict_row cursor."""

    def __init__(self, columns: dict[str, list[str]]) -> None:
        self.columns = columns
        self.statements: list[tuple[Any, Any]] = []
        self._pending_table: str | None = None

    async def execute(self, query: Any, params: Any = None) -> None:
        self.statements.append((query, params))
        if isinstance(query, str) and "information_schema.columns" in query:
            self._pending_table = params[0]

    async def fetchall(self) -> list[dict[str, str]]:
        table = self._pending_table
        return [{"column_name": c} for c in self.columns.get(table or "", [])]

    async def __aenter__(self) -> "_Cursor":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor

    async def __aenter__(self) -> "_Connection":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _Pool:
    def __init__(self, cursor: _Cursor) -> None:
        self._connection = _Connection(cursor)

    def connection(self) -> _Connection:
        return self._connection


def _inserts(cursor: _Cursor) -> list[tuple[Any, Any]]:
    return [(q, p) for q, p in cursor.statements if isinstance(q, sql.Composed)]


class TestCopyThreadCheckpoints:
    @pytest.mark.asyncio
    async def test_copies_every_checkpoint_table(self) -> None:
        cursor = _Cursor(_COLUMNS)

        await copy_thread_checkpoints(_Pool(cursor), "src", "dst")

        assert len(_inserts(cursor)) == len(_CHECKPOINT_TABLES)

    @pytest.mark.asyncio
    async def test_binds_target_then_source(self) -> None:
        """The projection placeholder takes the new id; the WHERE takes the old one."""
        cursor = _Cursor(_COLUMNS)

        await copy_thread_checkpoints(_Pool(cursor), "src", "dst")

        for _query, params in _inserts(cursor):
            assert params == ("dst", "src")

    @pytest.mark.asyncio
    async def test_uses_catalog_columns_not_a_hardcoded_list(self) -> None:
        """LangGraph adds columns across versions; a drifted table must still copy."""
        drifted = dict(_COLUMNS)
        drifted["checkpoint_writes"] = [*_COLUMNS["checkpoint_writes"], "task_path"]
        cursor = _Cursor(drifted)

        await copy_thread_checkpoints(_Pool(cursor), "src", "dst")

        writes = _inserts(cursor)[_CHECKPOINT_TABLES.index("checkpoint_writes")][0]
        assert "task_path" in writes.as_string(None)

    @pytest.mark.asyncio
    async def test_skips_a_table_that_does_not_exist(self) -> None:
        cursor = _Cursor({**_COLUMNS, "checkpoint_blobs": []})

        await copy_thread_checkpoints(_Pool(cursor), "src", "dst")

        assert len(_inserts(cursor)) == len(_CHECKPOINT_TABLES) - 1

    @pytest.mark.asyncio
    async def test_tuple_rows_are_tolerated(self) -> None:
        """A pool configured without dict_row yields positional rows."""

        class _TupleCursor(_Cursor):
            async def fetchall(self) -> list[tuple[str]]:  # type: ignore[override]
                return [(c,) for c in self.columns.get(self._pending_table or "", [])]

        cursor = _TupleCursor(_COLUMNS)

        await copy_thread_checkpoints(_Pool(cursor), "src", "dst")

        assert len(_inserts(cursor)) == len(_CHECKPOINT_TABLES)
