import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.db import get_connection, initialize_database
from api.repositories.events import list_events
from ingestion.expand_exposure import expand_exposure_event
from ingestion.ingest_text import ingest_text_event
from ingestion.normalize_event import NormalizationError, NormalizedEvent, normalize_event
from ml.insights import list_insights, recompute_insights

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


class RecomputeInsightsOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    pairs_evaluated: int
    insights_written: int


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

# user submits entry
@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn):
    # normalize/validate first so db inserts always recieve standardized shape
    try:
        normalized = normalize_event(payload)
    except NormalizationError as exc:
        payload_dict = payload.model_dump()
        # keep failed payload for later parsing/review in raw_event_ingest table
        _store_raw_event(payload.user_id, payload_dict, str(exc))
        return _event_response(-1, payload_dict, status="queued", resolution="pending")

    try:
        created_id = _insert_event_and_expand(normalized)
    except sqlite3.DatabaseError as exc:
        # db write failed, queue for retry/review instead of surfacing stack trace
        _store_raw_event(normalized["user_id"], normalized, str(exc))
        return _event_response(-1, normalized, status="queued", resolution="db_error")

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

# list insights per user
@app.get("/insights", response_model=list[InsightOut])
def get_insights(user_id: int, include_suppressed: bool = True):
    return list_insights(user_id=user_id, include_suppressed=include_suppressed)

# store event in loose ends table if ingestion failure
def _store_raw_event(user_id: int | None, payload, error: str):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO raw_event_ingest (user_id, raw_text, ingested_at, parse_status, error)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            json.dumps(payload),
            datetime.now(tz=timezone.utc).isoformat(),
            "failed",
            error,
        ),
    )
    conn.commit()
    conn.close()
