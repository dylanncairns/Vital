import json
import os
import sqlite3
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.db import get_connection, initialize_database
from api.repositories.events import list_events
from api.repositories.jobs import (
    JOB_EVIDENCE_ACQUIRE_CANDIDATE,
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
    list_pending_jobs,
    mark_job_done,
    mark_job_failed,
)
from api.repositories.raw_event_ingest import insert_raw_event_ingest
from ingestion.expand_exposure import expand_exposure_event
from ingestion.ingest_text import ingest_text_event
from ingestion.normalize_event import NormalizationError, NormalizedEvent, normalize_event
from ml.insights import list_insights, list_rag_sync_candidates, recompute_insights
from ml.rag import (
    fetch_ingredient_name_map,
    fetch_item_name_map,
    fetch_symptom_name_map,
    sync_claims_for_candidates,
)
from ml.vector_ingest import ingest_sources_for_candidates

app = FastAPI(
    title="Vital API",
    version="0.1.0",
)

# Ensure SQLite schema exists on startup
@app.on_event("startup")
def on_startup():
    initialize_database()

# Validate / standardize input JSON payload format for /events
# Will contain user_id and either item_id + route or symptom_id + severity
# supports id or name resolution fields, exact or range time fields, can be raw text
class EventIn(BaseModel):
    event_type: str = Field(pattern="^(exposure|symptom)$")
    user_id: int
    timestamp: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    time_confidence: Optional[str] = Field(default=None, pattern="^(exact|approx|backfilled)$")
    raw_text: Optional[str] = None
    item_id: int | None = None
    item_name: Optional[str] = None
    route: str | None = None
    symptom_id: int | None = None
    symptom_name: Optional[str] = None
    severity: int | None = None

# Validate / standardize output JSON payload format for /events - adds event id to enable echoing data after an insert
class EventOut(EventIn):
    id: int
    status: Optional[str] = None
    resolution: Optional[str] = None

# Define the JSON format for an entity in timeline (a timeline row that get/events returns)
class TimelineEvent(BaseModel):
    id: int
    event_type: str
    user_id: int
    timestamp: Optional[str] = None
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    route: Optional[str] = None
    symptom_id: Optional[int] = None
    symptom_name: Optional[str] = None
    severity: Optional[int] = None

# Shape of ingested text from user text blurb input 
class TextIngestIn(BaseModel):
    user_id: int
    raw_text: str

# Response shape for /events/ingest_text
class TextIngestOut(BaseModel):
    status: str
    event_type: Optional[str] = None
    resolution: Optional[str] = None
    reason: Optional[str] = None


class RecomputeInsightsIn(BaseModel):
    user_id: int
    online_enabled: bool = Field(
        default_factory=lambda: os.getenv("RAG_ENABLE_ONLINE_RETRIEVAL", "1") == "1"
    )
    max_papers_per_query: int = Field(default=3, ge=1, le=10)


class RecomputeInsightsOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    pairs_evaluated: int
    insights_written: int


class RagSyncIn(BaseModel):
    user_id: int
    online_enabled: bool = True
    max_papers_per_query: int = Field(default=3, ge=1, le=10)


class RagSyncOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    queries_built: int
    papers_added: int
    claims_added: int


class ProcessJobsIn(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
    max_papers_per_query: int = Field(default=3, ge=1, le=10)


class ProcessJobsOut(BaseModel):
    status: str
    jobs_claimed: int
    jobs_done: int
    jobs_failed: int
    recompute_jobs_done: int
    evidence_jobs_done: int


class InsightCitationOut(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    evidence_polarity_and_strength: Optional[int] = None


class InsightOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    item_name: str
    symptom_id: int
    symptom_name: str
    model_probability: Optional[float] = None
    evidence_strength_score: Optional[float] = None
    evidence_summary: Optional[str] = None
    display_decision_reason: Optional[str] = None
    display_status: Optional[str] = None
    created_at: Optional[str] = None
    citations: list[InsightCitationOut]


def _event_response(event_id: int, event: dict, status: str | None = None, resolution: str | None = None) -> dict:
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

def _insert_event_and_expand(normalized: NormalizedEvent) -> int:
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            created_id = cursor.lastrowid
            # insert exposure expansion rows before commit so writes are atomic
            expand_exposure_event(created_id, conn=conn)
        else:
            cursor = conn.execute(
                """
                INSERT INTO symptom_events (
                    user_id, symptom_id, timestamp, time_range_start, time_range_end,
                    time_confidence, ingested_at, raw_text, severity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            created_id = cursor.lastrowid

        conn.commit()
        return created_id
    except sqlite3.DatabaseError:
        conn.rollback()
        raise
    finally:
        conn.close()


def _enqueue_impacted_recompute_jobs(normalized: NormalizedEvent) -> int:
    user_id = int(normalized["user_id"])
    jobs_added = 0
    conn = get_connection()
    try:
        if normalized["event_type"] == "exposure":
            item_id = int(normalized["item_id"])
            symptom_rows = conn.execute(
                "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = ?",
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
                )
                if job_id is not None:
                    jobs_added += 1
        else:
            symptom_id = int(normalized["symptom_id"])
            item_rows = conn.execute(
                "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = ?",
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
                )
                if job_id is not None:
                    jobs_added += 1
        return jobs_added
    finally:
        conn.close()

# user submits entry
@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn):
    # normalize/validate first so db inserts always recieve standardized shape
    try:
        normalized = normalize_event(payload)
    except NormalizationError as exc:
        payload_dict = payload.model_dump()
        # keep failed payload for later parsing/review in raw_event_ingest table
        insert_raw_event_ingest(payload.user_id, json.dumps(payload_dict), "failed", str(exc))
        return _event_response(-1, payload_dict, status="queued", resolution="pending")

    try:
        created_id = _insert_event_and_expand(normalized)
    except sqlite3.DatabaseError as exc:
        # db write failed, queue for retry/review instead of surfacing stack trace
        insert_raw_event_ingest(normalized["user_id"], json.dumps(normalized), "failed", str(exc))
        return _event_response(-1, normalized, status="queued", resolution="db_error")

    _enqueue_impacted_recompute_jobs(normalized)
    return _event_response(created_id, normalized)


# timeline display
@app.get("/events", response_model=list[TimelineEvent])
def get_events(user_id: int):
    return list_events(user_id)

# empty text submission handled, else call to ingest_text_event
@app.post("/events/ingest_text", response_model=TextIngestOut)
def ingest_text(payload: TextIngestIn):
    if not payload.raw_text.strip():
        return {"status": "ignored", "reason": "empty_raw_text"}
    return ingest_text_event(payload.user_id, payload.raw_text)

# compute insights for user 
@app.post("/insights/recompute", response_model=RecomputeInsightsOut)
def recompute_user_insights(payload: RecomputeInsightsIn):
    result = recompute_insights(payload.user_id)
    return {
        "status": "ok",
        "user_id": payload.user_id,
        **result,
    }


@app.post("/rag/sync", response_model=RagSyncOut)
def sync_rag_evidence(payload: RagSyncIn):
    candidates = list_rag_sync_candidates(payload.user_id)
    conn = get_connection()
    try:
        result = sync_claims_for_candidates(
            conn,
            candidates=candidates,
            ingredient_name_map=fetch_ingredient_name_map(conn),
            symptom_name_map=fetch_symptom_name_map(conn),
            item_name_map=fetch_item_name_map(conn),
            online_enabled=payload.online_enabled,
            max_papers_per_query=payload.max_papers_per_query,
        )
        conn.commit()
    except sqlite3.DatabaseError:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "status": "ok",
        "user_id": payload.user_id,
        "candidates_considered": len(candidates),
        **result,
    }


@app.post("/jobs/process", response_model=ProcessJobsOut)
def process_background_jobs(payload: ProcessJobsIn):
    jobs = list_pending_jobs(limit=payload.limit)
    jobs_done = 0
    jobs_failed = 0
    recompute_jobs_done = 0
    evidence_jobs_done = 0

    for job in jobs:
        job_id = int(job["id"])
        user_id = int(job["user_id"])
        item_id = job.get("item_id")
        symptom_id = job.get("symptom_id")
        try:
            if job["job_type"] == JOB_RECOMPUTE_CANDIDATE:
                if item_id is None or symptom_id is None:
                    raise ValueError("recompute job missing item_id or symptom_id")
                recompute_insights(user_id, target_pairs={(int(item_id), int(symptom_id))})
                recompute_jobs_done += 1

                conn = get_connection()
                try:
                    row = conn.execute(
                        """
                        SELECT evidence_strength_score
                        FROM insights
                        WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (user_id, int(item_id), int(symptom_id)),
                    ).fetchone()
                finally:
                    conn.close()

                evidence_strength = float(row["evidence_strength_score"]) if row and row["evidence_strength_score"] is not None else 0.0
                if evidence_strength <= 0.0:
                    enqueue_background_job(
                        user_id=user_id,
                        job_type=JOB_EVIDENCE_ACQUIRE_CANDIDATE,
                        item_id=int(item_id),
                        symptom_id=int(symptom_id),
                        payload={"trigger": "insufficient_evidence"},
                    )

            elif job["job_type"] == JOB_EVIDENCE_ACQUIRE_CANDIDATE:
                if item_id is None or symptom_id is None:
                    raise ValueError("evidence job missing item_id or symptom_id")
                candidates = [
                    candidate
                    for candidate in list_rag_sync_candidates(user_id)
                    if int(candidate["item_id"]) == int(item_id) and int(candidate["symptom_id"]) == int(symptom_id)
                ]
                if candidates:
                    conn = get_connection()
                    try:
                        sync_result = sync_claims_for_candidates(
                            conn,
                            candidates=candidates,
                            ingredient_name_map=fetch_ingredient_name_map(conn),
                            symptom_name_map=fetch_symptom_name_map(conn),
                            item_name_map=fetch_item_name_map(conn),
                            online_enabled=True,
                            max_papers_per_query=payload.max_papers_per_query,
                        )
                        conn.commit()
                    except sqlite3.DatabaseError:
                        conn.rollback()
                        raise
                    finally:
                        conn.close()

                    if int(sync_result.get("claims_added", 0)) == 0:
                        ingest_sources_for_candidates(
                            candidates=candidates,
                            vector_store_id=os.getenv("RAG_VECTOR_STORE_ID"),
                            max_queries=6,
                            max_papers_per_query=max(3, payload.max_papers_per_query),
                        )
                        conn_retry = get_connection()
                        try:
                            sync_claims_for_candidates(
                                conn_retry,
                                candidates=candidates,
                                ingredient_name_map=fetch_ingredient_name_map(conn_retry),
                                symptom_name_map=fetch_symptom_name_map(conn_retry),
                                item_name_map=fetch_item_name_map(conn_retry),
                                online_enabled=True,
                                max_papers_per_query=max(3, payload.max_papers_per_query),
                            )
                            conn_retry.commit()
                        except sqlite3.DatabaseError:
                            conn_retry.rollback()
                            raise
                        finally:
                            conn_retry.close()
                recompute_insights(user_id, target_pairs={(int(item_id), int(symptom_id))})
                evidence_jobs_done += 1
            else:
                raise ValueError(f"unknown job_type {job['job_type']}")

            mark_job_done(job_id)
            jobs_done += 1
        except Exception as exc:
            mark_job_failed(job_id, str(exc))
            jobs_failed += 1

    return {
        "status": "ok",
        "jobs_claimed": len(jobs),
        "jobs_done": jobs_done,
        "jobs_failed": jobs_failed,
        "recompute_jobs_done": recompute_jobs_done,
        "evidence_jobs_done": evidence_jobs_done,
    }

# list insights per user
@app.get("/insights", response_model=list[InsightOut])
def get_insights(user_id: int, include_suppressed: bool = True):
    return list_insights(user_id=user_id, include_suppressed=include_suppressed)
