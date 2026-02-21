# recurring exposure logic
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from api.db import get_connection
from ingestion.expand_exposure import expand_exposure_event


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def list_recurring_rules(user_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.user_id,
                r.item_id,
                i.name AS item_name,
                r.route,
                r.start_at,
                r.interval_hours,
                r.time_confidence,
                r.is_active,
                r.last_generated_at,
                r.notes,
                r.created_at,
                r.updated_at
            FROM recurring_exposure_rules r
            JOIN items i ON i.id = r.item_id
            WHERE r.user_id = %s
            ORDER BY r.id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def create_recurring_rule(
    *,
    user_id: int,
    item_id: int,
    route: str,
    start_at: str,
    interval_hours: int,
    time_confidence: str = "approx",
    notes: str | None = None,
) -> int:
    conn = get_connection()
    try:
        now = _now_iso()
        cursor = conn.execute(
            """
            INSERT INTO recurring_exposure_rules (
                user_id, item_id, route, start_at, interval_hours, time_confidence,
                is_active, last_generated_at, notes, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, 1, NULL, %s, %s, %s)
            RETURNING id
            """,
            (
                int(user_id),
                int(item_id),
                route,
                start_at,
                int(interval_hours),
                time_confidence,
                notes,
                now,
                now,
            ),
        )
        conn.commit()
        return int(cursor.fetchone()["id"])
    finally:
        conn.close()


def update_recurring_rule(
    *,
    user_id: int,
    rule_id: int,
    route: str | None = None,
    start_at: str | None = None,
    interval_hours: int | None = None,
    time_confidence: str | None = None,
    is_active: bool | None = None,
    notes: str | None = None,
) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM recurring_exposure_rules WHERE id = %s AND user_id = %s",
            (int(rule_id), int(user_id)),
        ).fetchone()
        if row is None:
            return False
        updates: list[str] = []
        params: list[Any] = []
        if route is not None:
            updates.append("route = %s")
            params.append(route)
        if start_at is not None:
            updates.append("start_at = %s")
            params.append(start_at)
            updates.append("last_generated_at = NULL")
        if interval_hours is not None:
            updates.append("interval_hours = %s")
            params.append(int(interval_hours))
            updates.append("last_generated_at = NULL")
        if time_confidence is not None:
            updates.append("time_confidence = %s")
            params.append(time_confidence)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(1 if is_active else 0)
        if notes is not None:
            updates.append("notes = %s")
            params.append(notes)
        if not updates:
            return True
        updates.append("updated_at = %s")
        params.append(_now_iso())
        params.extend([int(rule_id), int(user_id)])
        conn.execute(
            f"""
            UPDATE recurring_exposure_rules
            SET {", ".join(updates)}
            WHERE id = %s AND user_id = %s
            """,
            tuple(params),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def delete_recurring_rule(*, user_id: int, rule_id: int) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM recurring_exposure_rules WHERE id = %s AND user_id = %s",
            (int(rule_id), int(user_id)),
        )
        conn.commit()
        return int(cursor.rowcount) > 0
    finally:
        conn.close()


def materialize_recurring_exposures(
    *,
    user_id: int,
    max_events_per_rule: int = 64,
) -> list[int]:
    conn = get_connection()
    inserted_item_ids: list[int] = []
    try:
        rules = conn.execute(
            """
            SELECT
                id, item_id, route, start_at, interval_hours, time_confidence, last_generated_at
            FROM recurring_exposure_rules
            WHERE user_id = %s AND is_active = 1
            ORDER BY id ASC
            """,
            (int(user_id),),
        ).fetchall()
        now_dt = datetime.now(tz=timezone.utc)
        now_iso = now_dt.isoformat()

        for rule in rules:
            start_dt = _parse_iso_utc(rule["start_at"])
            if start_dt is None:
                continue
            interval_hours = int(rule["interval_hours"] or 0)
            if interval_hours <= 0:
                continue
            step = timedelta(hours=interval_hours)

            last_generated_dt = _parse_iso_utc(rule["last_generated_at"])
            if last_generated_dt is None:
                next_dt = start_dt
            else:
                next_dt = last_generated_dt + step

            generated_count = 0
            newest_generated: datetime | None = None
            while next_dt <= now_dt and generated_count < max_events_per_rule:
                cursor = conn.execute(
                    """
                    INSERT INTO exposure_events (
                        user_id, item_id, timestamp, time_range_start, time_range_end,
                        time_confidence, ingested_at, raw_text, route
                    )
                    VALUES (%s, %s, %s, NULL, NULL, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(user_id),
                        int(rule["item_id"]),
                        next_dt.isoformat(),
                        rule["time_confidence"] or "approx",
                        now_iso,
                        f"recurring_rule:{int(rule['id'])}",
                        rule["route"] or "other",
                    ),
                )
                expand_exposure_event(int(cursor.fetchone()["id"]), conn=conn)
                inserted_item_ids.append(int(rule["item_id"]))
                newest_generated = next_dt
                generated_count += 1
                next_dt = next_dt + step

            if newest_generated is not None:
                conn.execute(
                    """
                    UPDATE recurring_exposure_rules
                    SET last_generated_at = %s, updated_at = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (
                        newest_generated.isoformat(),
                        now_iso,
                        int(rule["id"]),
                        int(user_id),
                    ),
                )
        conn.commit()
        return inserted_item_ids
    finally:
        conn.close()
