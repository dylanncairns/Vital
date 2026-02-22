from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from api.db import get_connection


JOB_RECOMPUTE_CANDIDATE = "recompute_candidate"
JOB_EVIDENCE_ACQUIRE_CANDIDATE = "evidence_acquire_candidate"
JOB_MODEL_RETRAIN = "model_retrain"
JOB_CITATION_AUDIT = "citation_audit"
DEFAULT_MAX_FAILED_ATTEMPTS = 5
DEFAULT_FAILED_RETRY_BASE_SECONDS = 30
DEFAULT_FAILED_RETRY_MAX_SECONDS = 600
DEFAULT_RETRAIN_VERIFICATION_DELTA = int(os.getenv("MODEL_RETRAIN_VERIFICATION_DELTA", "10"))
_SECRET_REDACTION_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", re.I), r"\1[REDACTED_TOKEN]"),
]


def _sanitize_error_message(message: str) -> str:
    text = str(message or "")
    for pattern, repl in _SECRET_REDACTION_PATTERNS:
        text = pattern.sub(repl, text)
    if "Incorrect API key provided" in text:
        text = re.sub(
            r"Incorrect API key provided:\s*[^\.]+",
            "Incorrect API key provided: [REDACTED_API_KEY]",
            text,
        )
    return text


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


def _retry_delay_seconds(attempts: int) -> int:
    # attempts is incremented only on failure; first retry waits base seconds.
    exponent = max(0, attempts - 1)
    raw = DEFAULT_FAILED_RETRY_BASE_SECONDS * (2 ** exponent)
    return min(DEFAULT_FAILED_RETRY_MAX_SECONDS, raw)


def _count_validated_linkage_events(conn) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM insight_verifications
        WHERE COALESCE(verified, 0) = 1
           OR COALESCE(rejected, 0) = 1
        """
    ).fetchone()
    return int(row["c"]) if row is not None else 0


def enqueue_background_job(
    *,
    user_id: int,
    job_type: str,
    item_id: int | None,
    symptom_id: int | None,
    payload: dict[str, Any] | None = None,
    conn=None,
) -> int | None:
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()
    try:
        existing = conn.execute(
            """
            SELECT id FROM background_jobs
            WHERE user_id = %s
              AND job_type = %s
              AND status IN ('pending', 'running')
              AND ((item_id IS NULL AND %s::bigint IS NULL) OR item_id = %s::bigint)
              AND ((symptom_id IS NULL AND %s::bigint IS NULL) OR symptom_id = %s::bigint)
            LIMIT 1
            """,
            (user_id, job_type, item_id, item_id, symptom_id, symptom_id),
        ).fetchone()
        if existing is not None:
            return None

        now_iso = _now_iso()
        cursor = conn.execute(
            """
            INSERT INTO background_jobs (
                user_id, job_type, item_id, symptom_id, payload_json,
                status, attempts, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                job_type,
                item_id,
                symptom_id,
                json.dumps(payload or {}, sort_keys=True),
                now_iso,
                now_iso,
            ),
        )
        if owns_connection:
            conn.commit()
        return int(cursor.fetchone()["id"])
    finally:
        if owns_connection:
            conn.close()

# Recover stale ("pending") running jobs so a bad job does not occupy worker forever
def list_pending_jobs(*, limit: int = 20) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        # Label jobs running > 10 min with no update as pending
        conn.execute(
            """
            UPDATE background_jobs
            SET status = 'pending', updated_at = %s
            WHERE status = 'running'
              AND updated_at IS NOT NULL
              AND updated_at::timestamptz < (NOW() - INTERVAL '10 minutes')
            """,
            (_now_iso(),),
        )
        # Fetch now pending rows
        rows = conn.execute(
            """
            SELECT id, user_id, job_type, item_id, symptom_id, payload_json, status, attempts, updated_at
            FROM background_jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

        selected_rows = list(rows)

        # Retry failed jobs automatically
        remaining = max(0, limit - len(selected_rows))
        if remaining > 0:
            failed_rows = conn.execute(
                """
                SELECT id, user_id, job_type, item_id, symptom_id, payload_json, status, attempts, updated_at
                FROM background_jobs
                WHERE status = 'failed'
                  AND attempts < %s
                ORDER BY updated_at ASC, created_at ASC, id ASC
                LIMIT %s
                """,
                (DEFAULT_MAX_FAILED_ATTEMPTS, max(remaining * 5, 25)),
            ).fetchall()
            now_dt = datetime.now(tz=timezone.utc)
            selected_ids = {int(row["id"]) for row in selected_rows}
            for row in failed_rows:
                if len(selected_rows) >= limit:
                    break
                row_id = int(row["id"])
                if row_id in selected_ids:
                    continue
                attempts = int(row["attempts"] or 0)
                updated_at_dt = _parse_iso_utc(row["updated_at"])
                if updated_at_dt is None:
                    selected_rows.append(row)
                    selected_ids.add(row_id)
                    continue
                elapsed_seconds = (now_dt - updated_at_dt).total_seconds()
                if elapsed_seconds >= _retry_delay_seconds(attempts):
                    selected_rows.append(row)
                    selected_ids.add(row_id)

        if not selected_rows:
            return []
        
        job_ids = [int(row["id"]) for row in selected_rows]
        now_iso = _now_iso()
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE background_jobs
                SET status = 'running', updated_at = %s
                WHERE id = %s
                """,
                [(now_iso, job_id) for job_id in job_ids],
            )
        conn.commit()

        out: list[dict[str, Any]] = []
        for row in selected_rows:
            payload = {}
            if row["payload_json"]:
                try:
                    decoded = json.loads(row["payload_json"])
                    if isinstance(decoded, dict):
                        payload = decoded
                except json.JSONDecodeError:
                    payload = {}
            out.append(
                {
                    "id": int(row["id"]),
                    "user_id": int(row["user_id"]),
                    "job_type": row["job_type"],
                    "item_id": int(row["item_id"]) if row["item_id"] is not None else None,
                    "symptom_id": int(row["symptom_id"]) if row["symptom_id"] is not None else None,
                    "payload": payload,
                    "attempts": int(row["attempts"]),
                }
            )
        return out
    finally:
        conn.close()


def mark_job_done(job_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE background_jobs
            SET status = 'done', updated_at = %s
            WHERE id = %s
            """,
            (_now_iso(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_job_failed(job_id: int, error: str) -> None:
    conn = get_connection()
    try:
        safe_error = _sanitize_error_message(error)
        conn.execute(
            """
            UPDATE background_jobs
            SET status = 'failed',
                attempts = attempts + 1,
                last_error = %s,
                updated_at = %s
            WHERE id = %s
            """,
            (safe_error[:500], _now_iso(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def count_jobs(*, user_id: int | None = None, status: str | None = None) -> int:
    conn = get_connection()
    try:
        sql = "SELECT COUNT(*) AS c FROM background_jobs WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            sql += " AND user_id = %s"
            params.append(user_id)
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        row = conn.execute(sql, tuple(params)).fetchone()
        return int(row["c"]) if row is not None else 0
    finally:
        conn.close()


def maybe_enqueue_model_retrain(
    *,
    trigger_user_id: int,
    verification_delta_threshold: int = DEFAULT_RETRAIN_VERIFICATION_DELTA,
) -> int | None:
    conn = get_connection()
    try:
        existing = conn.execute(
            """
            SELECT id FROM background_jobs
            WHERE job_type = %s
              AND status IN ('pending', 'running')
            LIMIT 1
            """,
            (JOB_MODEL_RETRAIN,),
        ).fetchone()
        if existing is not None:
            return None

        total_validated_labels = _count_validated_linkage_events(conn)

        state = conn.execute(
            """
            SELECT last_trained_total_events, last_enqueued_total_events
            FROM model_retrain_state
            WHERE id = 1
            """
        ).fetchone()
        last_trained = int(state["last_trained_total_events"]) if state is not None else 0
        last_enqueued = int(state["last_enqueued_total_events"]) if state is not None else 0
        # Backward-compatible schema: these columns now track validated-label counts.
        # Clamp stale event-based counters so deployment upgrades start retraining again.
        if last_trained > total_validated_labels:
            last_trained = total_validated_labels
        if last_enqueued > total_validated_labels:
            last_enqueued = total_validated_labels
        baseline = last_trained
        if total_validated_labels - baseline < max(1, int(verification_delta_threshold)):
            return None

        now_iso = _now_iso()
        payload = {
            "trigger": "verification_delta_threshold",
            "total_validated_labels": total_validated_labels,
            "last_trained_total_labels": last_trained,
            "verification_delta_threshold": int(verification_delta_threshold),
        }
        cursor = conn.execute(
            """
            INSERT INTO background_jobs (
                user_id, job_type, item_id, symptom_id, payload_json,
                status, attempts, created_at, updated_at
            )
            VALUES (%s, %s, NULL, NULL, %s, 'pending', 0, %s, %s)
            RETURNING id
            """,
            (
                int(trigger_user_id),
                JOB_MODEL_RETRAIN,
                json.dumps(payload, sort_keys=True),
                now_iso,
                now_iso,
            ),
        )
        conn.execute(
            """
            UPDATE model_retrain_state
            SET last_enqueued_total_events = %s, updated_at = %s
            WHERE id = 1
            """,
            (total_validated_labels, now_iso),
        )
        conn.commit()
        return int(cursor.fetchone()["id"])
    finally:
        conn.close()


def mark_model_retrain_completed(*, trained_total_events: int | None = None) -> None:
    conn = get_connection()
    try:
        if trained_total_events is None:
            trained_total_events = _count_validated_linkage_events(conn)
        conn.execute(
            """
            UPDATE model_retrain_state
            SET last_trained_total_events = %s,
                last_enqueued_total_events = %s,
                updated_at = %s
            WHERE id = 1
            """,
            (int(trained_total_events), int(trained_total_events), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
