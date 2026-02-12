from __future__ import annotations

from datetime import datetime, timezone

from api.db import get_connection


def insert_raw_event_ingest(
    user_id: int | None,
    raw_text: str,
    parse_status: str,
    error: str | None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO raw_event_ingest (user_id, raw_text, ingested_at, parse_status, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                raw_text,
                datetime.now(tz=timezone.utc).isoformat(),
                parse_status,
                error,
            ),
        )
        conn.commit()
    finally:
        conn.close()
