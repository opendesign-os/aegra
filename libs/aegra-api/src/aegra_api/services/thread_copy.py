"""Checkpoint-lineage copy backing ``POST /threads/{thread_id}/copy``.

``BaseCheckpointSaver.acopy_thread`` is an unimplemented stub on the Postgres
saver, so the copy is done here. Column lists come from the catalog rather
than being hardcoded: LangGraph adds columns to these tables across versions.
"""

from __future__ import annotations

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

_CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


async def copy_thread_checkpoints(
    pool: AsyncConnectionPool,
    source_thread_id: str,
    target_thread_id: str,
) -> None:
    async with pool.connection() as conn, conn.cursor() as cur:
        for table in _CHECKPOINT_TABLES:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
                (table,),
            )
            columns = [row["column_name"] if isinstance(row, dict) else row[0] for row in await cur.fetchall()]
            if not columns:
                continue

            targets = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
            projection = sql.SQL(", ").join(
                sql.Placeholder() if c == "thread_id" else sql.Identifier(c) for c in columns
            )
            await cur.execute(
                sql.SQL("INSERT INTO {table} ({targets}) SELECT {projection} FROM {table} WHERE thread_id = %s").format(
                    table=sql.Identifier(table), targets=targets, projection=projection
                ),
                (target_thread_id, source_thread_id),
            )
