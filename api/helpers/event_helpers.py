from __future__ import annotations

from psycopg import Error as DatabaseError

from api.db import get_connection
from api.worker.jobs import (
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
)
from ingestion.expand_exposure import expand_exposure_event
from ingestion.normalize_event import NormalizedEvent
from ingestion.time_utils import to_utc_iso


def normalize_patch_time_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = to_utc_iso(value, strict=True)
    if normalized is None:
        return None
    return normalized


def event_response(
    event_id: int,
    event: dict,
    status: str | None = None,
    resolution: str | None = None,
) -> dict:
    # one response shape for both successful writes and queued failures
    return {
        "id": event_id,
        "event_type": event.get("event_type"),
        "user_id": event.get("user_id"),
        "timestamp": event.get("timestamp"),
        "time_range_start": event.get("time_range_start"),
        "time_range_end": event.get("time_range_end"),
        "time_confidence": event.get("time_confidence"),
        "raw_text": event.get("raw_text"),
        "item_id": event.get("item_id"),
        "route": event.get("route"),
        "symptom_id": event.get("symptom_id"),
        "severity": event.get("severity"),
        "status": status,
        "resolution": resolution,
    }


def insert_event_and_expand(normalized: NormalizedEvent) -> int:
    # one db transaction for insert + exposure expansion
    conn = get_connection()
    try:
        if normalized["event_type"] == "exposure":
            cursor = conn.execute(
                """
                INSERT INTO exposure_events (
                    user_id, item_id, timestamp, time_range_start, time_range_end,
                    time_confidence, ingested_at, raw_text, route
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    normalized["user_id"],
                    normalized["item_id"],
                    normalized["timestamp"],
                    normalized["time_range_start"],
                    normalized["time_range_end"],
                    normalized["time_confidence"],
                    normalized["ingested_at"],
                    normalized["raw_text"],
                    normalized["route"],
                ),
            )
            created_row = cursor.fetchone()
            created_id = int(created_row["id"])
            # insert exposure expansion rows before commit so writes are atomic
            expand_exposure_event(created_id, conn=conn)
        else:
            cursor = conn.execute(
                """
                INSERT INTO symptom_events (
                    user_id, symptom_id, timestamp, time_range_start, time_range_end,
                    time_confidence, ingested_at, raw_text, severity
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    normalized["user_id"],
                    normalized["symptom_id"],
                    normalized["timestamp"],
                    normalized["time_range_start"],
                    normalized["time_range_end"],
                    normalized["time_confidence"],
                    normalized["ingested_at"],
                    normalized["raw_text"],
                    normalized["severity"],
                ),
            )
            created_row = cursor.fetchone()
            created_id = int(created_row["id"])

        conn.commit()
        return created_id
    except DatabaseError:
        conn.rollback()
        raise
    finally:
        conn.close()

# when event added, recompute opposite (symptom vs exposure) candidates with new entry info included
def enqueue_impacted_recompute_jobs(normalized: NormalizedEvent) -> int:
    user_id = int(normalized["user_id"])
    jobs_added = 0
    conn = get_connection()
    try:
        if normalized["event_type"] == "exposure":
            item_id = int(normalized["item_id"])
            symptom_rows = conn.execute(
                "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = %s",
                (user_id,),
            ).fetchall()
            symptom_ids = [int(row["symptom_id"]) for row in symptom_rows if row["symptom_id"] is not None]
            if not symptom_ids:
                return 0
            for symptom_id in symptom_ids:
                job_id = enqueue_background_job(
                    user_id=user_id,
                    job_type=JOB_RECOMPUTE_CANDIDATE,
                    item_id=item_id,
                    symptom_id=symptom_id,
                    payload={"trigger": "event_exposure"},
                    conn=conn,
                )
                if job_id is not None:
                    jobs_added += 1
        else:
            symptom_id = int(normalized["symptom_id"])
            item_rows = conn.execute(
                "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = %s",
                (user_id,),
            ).fetchall()
            item_ids = [int(row["item_id"]) for row in item_rows if row["item_id"] is not None]
            if not item_ids:
                return 0
            for item_id in item_ids:
                job_id = enqueue_background_job(
                    user_id=user_id,
                    job_type=JOB_RECOMPUTE_CANDIDATE,
                    item_id=item_id,
                    symptom_id=symptom_id,
                    payload={"trigger": "event_symptom"},
                    conn=conn,
                )
                if job_id is not None:
                    jobs_added += 1
        if jobs_added > 0:
            conn.commit()
        return jobs_added
    finally:
        conn.close()


def enqueue_recompute_jobs_for_items(*, user_id: int, item_ids: set[int], trigger: str) -> int:
    if not item_ids:
        return 0
    conn = get_connection()
    jobs_added = 0
    try:
        symptom_rows = conn.execute(
            "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = %s",
            (int(user_id),),
        ).fetchall()
        symptom_ids = [int(row["symptom_id"]) for row in symptom_rows if row["symptom_id"] is not None]
        for item_id in sorted(item_ids):
            for symptom_id in symptom_ids:
                job_id = enqueue_background_job(
                    user_id=int(user_id),
                    job_type=JOB_RECOMPUTE_CANDIDATE,
                    item_id=int(item_id),
                    symptom_id=int(symptom_id),
                    payload={"trigger": trigger},
                    conn=conn,
                )
                if job_id is not None:
                    jobs_added += 1
        if jobs_added > 0:
            conn.commit()
        return jobs_added
    finally:
        conn.close()


def enqueue_recompute_jobs_for_symptoms(*, user_id: int, symptom_ids: set[int], trigger: str) -> int:
    if not symptom_ids:
        return 0
    conn = get_connection()
    jobs_added = 0
    try:
        item_rows = conn.execute(
            "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = %s",
            (int(user_id),),
        ).fetchall()
        item_ids = [int(row["item_id"]) for row in item_rows if row["item_id"] is not None]
        for symptom_id in sorted(symptom_ids):
            for item_id in item_ids:
                job_id = enqueue_background_job(
                    user_id=int(user_id),
                    job_type=JOB_RECOMPUTE_CANDIDATE,
                    item_id=int(item_id),
                    symptom_id=int(symptom_id),
                    payload={"trigger": trigger},
                    conn=conn,
                )
                if job_id is not None:
                    jobs_added += 1
        if jobs_added > 0:
            conn.commit()
        return jobs_added
    finally:
        conn.close()
