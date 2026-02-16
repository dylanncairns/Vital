import json
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.db import get_connection, initialize_database
from api.repositories.events import list_events
from api.repositories.auth import (
    create_user,
    login_user,
    resolve_user_from_token,
    revoke_session,
)
from api.repositories.jobs import (
    JOB_CITATION_AUDIT,
    JOB_EVIDENCE_ACQUIRE_CANDIDATE,
    JOB_MODEL_RETRAIN,
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
    list_pending_jobs,
    mark_model_retrain_completed,
    mark_job_done,
    mark_job_failed,
    maybe_enqueue_model_retrain,
)
from ml.citation_audit import audit_claim_citations
from api.repositories.raw_event_ingest import insert_raw_event_ingest
from api.repositories.resolve import resolve_item_id, resolve_symptom_id
from api.repositories.recurring import (
    create_recurring_rule,
    delete_recurring_rule,
    list_recurring_rules,
    materialize_recurring_exposures,
    update_recurring_rule,
)
from ingestion.expand_exposure import expand_exposure_event
from ingestion.ingest_text import ingest_text_event
from ingestion.normalize_event import (
    NormalizationError,
    NormalizedEvent,
    normalize_event,
    normalize_route,
)
from ingestion.time_utils import to_utc_iso
from ml.insights import (
    list_event_insight_links,
    list_insights,
    list_rag_sync_candidates,
    recompute_insights,
    set_insight_rejection,
    set_insight_verification,
)
from ml.rag import (
    fetch_ingredient_name_map,
    fetch_item_name_map,
    fetch_symptom_name_map,
    sync_claims_for_candidates,
)
from ml.vector_ingest import ingest_sources_for_candidates
from ml.training_pipeline import run_training


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _ = app
    initialize_database()
    yield


app = FastAPI(
    title="Vital API",
    version="0.1.0",
    lifespan=_lifespan,
)

def _cors_allow_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    if configured.strip():
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
    ]

_cors_allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or r"^https://.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_origin_regex=_cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate / standardize input JSON payload format for /events
# Will contain user_id and either item_id + route or symptom_id + severity
# supports id or name resolution fields, exact or range time fields, can be raw text
class EventIn(BaseModel):
    event_type: str = Field(pattern="^(exposure|symptom)$")
    user_id: Optional[int] = None
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
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    time_confidence: Optional[str] = None
    raw_text: Optional[str] = None
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    route: Optional[str] = None
    symptom_id: Optional[int] = None
    symptom_name: Optional[str] = None
    severity: Optional[int] = None

# Shape of ingested text from user text blurb input 
class TextIngestIn(BaseModel):
    user_id: Optional[int] = None
    raw_text: str
    timezone_offset_minutes: Optional[int] = Field(default=None, ge=-840, le=840)

# Response shape for /events/ingest_text
class TextIngestOut(BaseModel):
    status: str
    event_type: Optional[str] = None
    resolution: Optional[str] = None
    reason: Optional[str] = None


class RecomputeInsightsIn(BaseModel):
    user_id: Optional[int] = None
    online_enabled: bool = Field(
        default_factory=lambda: os.getenv("RAG_ENABLE_ONLINE_RETRIEVAL", "1") == "1"
    )
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class RecomputeInsightsOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    pairs_evaluated: int
    insights_written: int


class RagSyncIn(BaseModel):
    user_id: Optional[int] = None
    online_enabled: bool = True
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class RagSyncOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    queries_built: int
    papers_added: int
    claims_added: int


class ProcessJobsIn(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class ProcessJobsOut(BaseModel):
    status: str
    jobs_claimed: int
    jobs_done: int
    jobs_failed: int
    recompute_jobs_done: int
    evidence_jobs_done: int
    model_retrain_jobs_done: int
    citation_audit_jobs_done: int


class CitationAuditIn(BaseModel):
    user_id: Optional[int] = None
    limit: int = Field(default=300, ge=1, le=5000)
    delete_missing: bool = True


class CitationAuditOut(BaseModel):
    status: str
    scanned_urls: int
    missing_urls: int
    deleted_claims: int
    deleted_papers: int
    errors: int


class CitationAuditEnqueueOut(BaseModel):
    status: str
    job_id: Optional[int] = None


class EventPatchIn(BaseModel):
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


class EventMutationOut(BaseModel):
    status: str
    event_id: int
    event_type: str
    jobs_queued: int


class RecurringExposureRuleIn(BaseModel):
    user_id: Optional[int] = None
    item_id: int | None = None
    item_name: Optional[str] = None
    route: str = "ingestion"
    start_at: str
    interval_hours: int = Field(ge=1, le=24 * 30)
    time_confidence: str = Field(default="approx", pattern="^(exact|approx|backfilled)$")
    notes: Optional[str] = None


class RecurringExposureRulePatchIn(BaseModel):
    route: str | None = None
    start_at: str | None = None
    interval_hours: int | None = Field(default=None, ge=1, le=24 * 30)
    time_confidence: str | None = Field(default=None, pattern="^(exact|approx|backfilled)$")
    is_active: bool | None = None
    notes: Optional[str] = None


class RecurringExposureRuleOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    item_name: str
    route: str
    start_at: str
    interval_hours: int
    time_confidence: Optional[str] = None
    is_active: int
    last_generated_at: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class InsightCitationOut(BaseModel):
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    evidence_polarity_and_strength: Optional[float] = None


class InsightOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    item_name: str
    secondary_item_id: Optional[int] = None
    secondary_item_name: Optional[str] = None
    is_combo: bool = False
    combo_key: Optional[str] = None
    combo_item_ids: Optional[list[int]] = None
    source_ingredient_id: Optional[int] = None
    source_ingredient_name: Optional[str] = None
    symptom_id: int
    symptom_name: str
    model_probability: Optional[float] = None
    evidence_support_score: Optional[float] = None
    evidence_strength_score: Optional[float] = None
    evidence_quality_score: Optional[float] = None
    penalty_score: Optional[float] = None
    overall_confidence_score: Optional[float] = None
    evidence_summary: Optional[str] = None
    display_decision_reason: Optional[str] = None
    display_status: Optional[str] = None
    user_verified: bool = False
    user_rejected: bool = False
    created_at: Optional[str] = None
    citations: list[InsightCitationOut]


class EventInsightLinkOut(BaseModel):
    event_type: str
    event_id: int
    insight_id: int


class InsightVerifyIn(BaseModel):
    user_id: Optional[int] = None
    verified: bool


class InsightVerifyOut(BaseModel):
    status: str
    insight_id: int
    user_id: int
    item_id: int
    symptom_id: int
    verified: bool
    rejected: bool = False


class InsightRejectIn(BaseModel):
    user_id: Optional[int] = None
    rejected: bool


class AuthRegisterIn(BaseModel):
    username: str
    password: str
    name: Optional[str] = None


class AuthLoginIn(BaseModel):
    username: str
    password: str


class AuthUserOut(BaseModel):
    id: int
    username: str
    name: Optional[str] = None


class AuthTokenOut(BaseModel):
    token: str
    user: AuthUserOut


class AuthMePatchIn(BaseModel):
    name: str


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if authorization is None:
        return None
    if not isinstance(authorization, str):
        return None
    value = authorization.strip()
    if not value:
        return None
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        return token or None
    return None


def _resolve_request_user_id(
    *,
    explicit_user_id: Optional[int],
    authorization: Optional[str],
    allow_legacy_explicit: bool = True,
) -> int:
    token = _extract_bearer_token(authorization)
    auth_user = resolve_user_from_token(token) if token else None
    if auth_user:
        auth_user_id = int(auth_user["id"])
        if explicit_user_id is not None and int(explicit_user_id) != auth_user_id:
            raise HTTPException(status_code=403, detail="user_id does not match auth token")
        return auth_user_id
    if allow_legacy_explicit and explicit_user_id is not None:
        return int(explicit_user_id)
    raise HTTPException(status_code=401, detail="Authentication required")


def _normalize_patch_time_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = to_utc_iso(value, strict=True)
    if normalized is None:
        return None
    return normalized


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


def _enqueue_recompute_jobs_for_items(*, user_id: int, item_ids: set[int], trigger: str) -> int:
    if not item_ids:
        return 0
    conn = get_connection()
    jobs_added = 0
    try:
        symptom_rows = conn.execute(
            "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = ?",
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
                )
                if job_id is not None:
                    jobs_added += 1
        return jobs_added
    finally:
        conn.close()


def _enqueue_recompute_jobs_for_symptoms(*, user_id: int, symptom_ids: set[int], trigger: str) -> int:
    if not symptom_ids:
        return 0
    conn = get_connection()
    jobs_added = 0
    try:
        item_rows = conn.execute(
            "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = ?",
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
                )
                if job_id is not None:
                    jobs_added += 1
        return jobs_added
    finally:
        conn.close()


@app.post("/auth/register", response_model=AuthTokenOut)
def register_auth(payload: AuthRegisterIn):
    try:
        created = create_user(username=payload.username, password=payload.password, name=payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    auth = login_user(username=payload.username, password=payload.password)
    if auth is None:
        raise HTTPException(status_code=500, detail="registration succeeded but login failed")
    return auth


@app.post("/auth/login", response_model=AuthTokenOut)
def login_auth(payload: AuthLoginIn):
    auth = login_user(username=payload.username, password=payload.password)
    if auth is None:
        raise HTTPException(status_code=401, detail="invalid username or password")
    return auth


@app.post("/auth/logout")
def logout_auth(authorization: Optional[str] = Header(default=None)):
    revoke_session(_extract_bearer_token(authorization))
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthUserOut)
def auth_me(authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)
    auth_user = resolve_user_from_token(token)
    if auth_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {
        "id": int(auth_user["id"]),
        "username": str(auth_user.get("username") or ""),
        "name": auth_user.get("name"),
    }


@app.patch("/auth/me", response_model=AuthUserOut)
def auth_me_patch(payload: AuthMePatchIn, authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)
    auth_user = resolve_user_from_token(token)
    if auth_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    next_name = (payload.name or "").strip()
    if len(next_name) < 2:
        raise HTTPException(status_code=400, detail="name must be at least 2 characters")
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET name = ? WHERE id = ?",
            (next_name, int(auth_user["id"])),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": int(auth_user["id"]),
        "username": str(auth_user.get("username") or ""),
        "name": next_name,
    }


# user submits entry
@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn, authorization: Optional[str] = Header(default=None)):
    payload = payload.model_copy(
        update={
            "user_id": _resolve_request_user_id(
                explicit_user_id=payload.user_id,
                authorization=authorization,
                allow_legacy_explicit=True,
            )
        }
    )
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
    maybe_enqueue_model_retrain(trigger_user_id=int(normalized["user_id"]))
    return _event_response(created_id, normalized)


@app.patch("/events/{event_type}/{event_id}", response_model=EventMutationOut)
def patch_event(
    event_type: str,
    event_id: int,
    payload: EventPatchIn,
    user_id: Optional[int] = None,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    normalized_type = event_type.strip().lower()
    if normalized_type not in {"exposure", "symptom"}:
        raise HTTPException(status_code=400, detail="event_type must be exposure or symptom")

    conn = get_connection()
    try:
        if normalized_type == "exposure":
            existing = conn.execute(
                "SELECT id, item_id FROM exposure_events WHERE id = ? AND user_id = ?",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            updates: list[str] = []
            params: list[object] = []
            resolved_item_id = payload.item_id
            if resolved_item_id is None and payload.item_name:
                resolved_item_id = resolve_item_id(payload.item_name)
            if resolved_item_id is not None:
                updates.append("item_id = ?")
                params.append(int(resolved_item_id))
            if payload.route is not None:
                updates.append("route = ?")
                try:
                    normalized_route = normalize_route(payload.route, strict=True)
                except NormalizationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                params.append(normalized_route)
            if payload.timestamp is not None:
                updates.append("timestamp = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.timestamp))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: timestamp")
            if payload.time_range_start is not None:
                updates.append("time_range_start = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.time_range_start))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_start")
            if payload.time_range_end is not None:
                updates.append("time_range_end = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.time_range_end))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_end")
            if payload.time_confidence is not None:
                updates.append("time_confidence = ?")
                params.append(payload.time_confidence)
            if payload.raw_text is not None:
                updates.append("raw_text = ?")
                params.append(payload.raw_text)
            if updates:
                params.extend([int(event_id), int(user_id)])
                conn.execute(
                    f"UPDATE exposure_events SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                    tuple(params),
                )
                conn.execute("DELETE FROM exposure_expansions WHERE exposure_event_id = ?", (int(event_id),))
                expand_exposure_event(int(event_id), conn=conn)
            conn.commit()
            old_item_id = int(existing["item_id"])
            new_item_id = int(resolved_item_id) if resolved_item_id is not None else old_item_id
            jobs_queued = _enqueue_recompute_jobs_for_items(
                user_id=int(user_id),
                item_ids={old_item_id, new_item_id},
                trigger="event_patch_exposure",
            )
        else:
            existing = conn.execute(
                "SELECT id, symptom_id FROM symptom_events WHERE id = ? AND user_id = ?",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            updates = []
            params = []
            resolved_symptom_id = payload.symptom_id
            if resolved_symptom_id is None and payload.symptom_name:
                resolved_symptom_id = resolve_symptom_id(payload.symptom_name)
            if resolved_symptom_id is not None:
                updates.append("symptom_id = ?")
                params.append(int(resolved_symptom_id))
            if payload.severity is not None:
                updates.append("severity = ?")
                params.append(int(payload.severity))
            if payload.timestamp is not None:
                updates.append("timestamp = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.timestamp))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: timestamp")
            if payload.time_range_start is not None:
                updates.append("time_range_start = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.time_range_start))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_start")
            if payload.time_range_end is not None:
                updates.append("time_range_end = ?")
                try:
                    params.append(_normalize_patch_time_value(payload.time_range_end))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_end")
            if payload.time_confidence is not None:
                updates.append("time_confidence = ?")
                params.append(payload.time_confidence)
            if payload.raw_text is not None:
                updates.append("raw_text = ?")
                params.append(payload.raw_text)
            if updates:
                params.extend([int(event_id), int(user_id)])
                conn.execute(
                    f"UPDATE symptom_events SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                    tuple(params),
                )
            conn.commit()
            old_symptom_id = int(existing["symptom_id"])
            new_symptom_id = int(resolved_symptom_id) if resolved_symptom_id is not None else old_symptom_id
            jobs_queued = _enqueue_recompute_jobs_for_symptoms(
                user_id=int(user_id),
                symptom_ids={old_symptom_id, new_symptom_id},
                trigger="event_patch_symptom",
            )

        maybe_enqueue_model_retrain(trigger_user_id=int(user_id))
        return {"status": "ok", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": int(jobs_queued)}
    finally:
        conn.close()


@app.delete("/events/{event_type}/{event_id}", response_model=EventMutationOut)
def delete_event(
    event_type: str,
    event_id: int,
    user_id: Optional[int] = None,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    normalized_type = event_type.strip().lower()
    if normalized_type not in {"exposure", "symptom"}:
        raise HTTPException(status_code=400, detail="event_type must be exposure or symptom")
    conn = get_connection()
    try:
        if normalized_type == "exposure":
            existing = conn.execute(
                "SELECT item_id FROM exposure_events WHERE id = ? AND user_id = ?",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            item_id = int(existing["item_id"])
            conn.execute("DELETE FROM exposure_expansions WHERE exposure_event_id = ?", (int(event_id),))
            conn.execute("DELETE FROM exposure_events WHERE id = ? AND user_id = ?", (int(event_id), int(user_id)))
            conn.commit()
            jobs_queued = _enqueue_recompute_jobs_for_items(
                user_id=int(user_id),
                item_ids={item_id},
                trigger="event_delete_exposure",
            )
        else:
            existing = conn.execute(
                "SELECT symptom_id FROM symptom_events WHERE id = ? AND user_id = ?",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            symptom_id = int(existing["symptom_id"])
            conn.execute("DELETE FROM symptom_events WHERE id = ? AND user_id = ?", (int(event_id), int(user_id)))
            conn.commit()
            jobs_queued = _enqueue_recompute_jobs_for_symptoms(
                user_id=int(user_id),
                symptom_ids={symptom_id},
                trigger="event_delete_symptom",
            )
        maybe_enqueue_model_retrain(trigger_user_id=int(user_id))
        return {"status": "ok", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": int(jobs_queued)}
    finally:
        conn.close()


@app.get("/recurring_exposures", response_model=list[RecurringExposureRuleOut])
def get_recurring_exposures(user_id: Optional[int] = None, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    return list_recurring_rules(int(user_id))


@app.post("/recurring_exposures", response_model=RecurringExposureRuleOut)
def create_recurring_exposure(payload: RecurringExposureRuleIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    item_id = payload.item_id
    if item_id is None and payload.item_name:
        item_id = resolve_item_id(payload.item_name)
    if item_id is None:
        raise HTTPException(status_code=400, detail="item_id or item_name is required")
    try:
        normalized_rule_route = normalize_route(payload.route, strict=True) or "other"
    except NormalizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rule_id = create_recurring_rule(
        user_id=int(user_id),
        item_id=int(item_id),
        route=normalized_rule_route,
        start_at=payload.start_at,
        interval_hours=int(payload.interval_hours),
        time_confidence=payload.time_confidence,
        notes=payload.notes,
    )
    rows = list_recurring_rules(int(user_id))
    for row in rows:
        if int(row["id"]) == int(rule_id):
            return row
    raise HTTPException(status_code=500, detail="failed to create recurring rule")


@app.patch("/recurring_exposures/{rule_id}", response_model=RecurringExposureRuleOut)
def patch_recurring_exposure(
    rule_id: int,
    payload: RecurringExposureRulePatchIn,
    user_id: Optional[int] = None,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    normalized_patch_route = None
    if payload.route is not None:
        try:
            normalized_patch_route = normalize_route(payload.route, strict=True)
        except NormalizationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    updated = update_recurring_rule(
        user_id=int(user_id),
        rule_id=int(rule_id),
        route=normalized_patch_route,
        start_at=payload.start_at,
        interval_hours=payload.interval_hours,
        time_confidence=payload.time_confidence,
        is_active=payload.is_active,
        notes=payload.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="rule not found")
    rows = list_recurring_rules(int(user_id))
    for row in rows:
        if int(row["id"]) == int(rule_id):
            return row
    raise HTTPException(status_code=404, detail="rule not found")


@app.delete("/recurring_exposures/{rule_id}")
def remove_recurring_exposure(rule_id: int, user_id: Optional[int] = None, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    deleted = delete_recurring_rule(user_id=int(user_id), rule_id=int(rule_id))
    return {"status": "ok" if deleted else "not_found", "rule_id": int(rule_id)}


# timeline display
@app.get("/events", response_model=list[TimelineEvent])
def get_events(user_id: Optional[int] = None, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    inserted_item_ids = materialize_recurring_exposures(user_id=int(user_id))
    if inserted_item_ids:
        _enqueue_recompute_jobs_for_items(
            user_id=int(user_id),
            item_ids=set(int(v) for v in inserted_item_ids),
            trigger="recurring_materialize",
        )
    return list_events(user_id)

# empty text submission handled, else call to ingest_text_event
@app.post("/events/ingest_text", response_model=TextIngestOut)
def ingest_text(payload: TextIngestIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    if not payload.raw_text.strip():
        return {"status": "ignored", "reason": "empty_raw_text"}
    return ingest_text_event(
        user_id,
        payload.raw_text,
        tz_offset_minutes=payload.timezone_offset_minutes,
    )

# compute insights for user 
@app.post("/insights/recompute", response_model=RecomputeInsightsOut)
def recompute_user_insights(payload: RecomputeInsightsIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    result = recompute_insights(user_id)
    return {
        "status": "ok",
        "user_id": user_id,
        **result,
    }


@app.post("/rag/sync", response_model=RagSyncOut)
def sync_rag_evidence(payload: RagSyncIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    candidates = list_rag_sync_candidates(user_id)
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
        "user_id": user_id,
        "candidates_considered": len(candidates),
        **result,
    }


@app.post("/citations/audit", response_model=CitationAuditOut)
def audit_citations(payload: CitationAuditIn):
    conn = get_connection()
    try:
        result = audit_claim_citations(
            conn,
            limit=payload.limit,
            delete_missing=payload.delete_missing,
        )
        conn.commit()
    except sqlite3.DatabaseError:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"status": "ok", **result}


@app.post("/citations/audit/enqueue", response_model=CitationAuditEnqueueOut)
def enqueue_citation_audit(payload: CitationAuditIn):
    enqueue_user_id = int(payload.user_id) if payload.user_id is not None else 1
    job_id = enqueue_background_job(
        user_id=enqueue_user_id,
        job_type=JOB_CITATION_AUDIT,
        item_id=None,
        symptom_id=None,
        payload={"limit": int(payload.limit), "delete_missing": bool(payload.delete_missing)},
    )
    return {"status": "ok", "job_id": job_id}


@app.post("/jobs/process", response_model=ProcessJobsOut)
def process_background_jobs(payload: ProcessJobsIn):
    jobs = list_pending_jobs(limit=payload.limit)
    jobs_done = 0
    jobs_failed = 0
    recompute_jobs_done = 0
    evidence_jobs_done = 0
    model_retrain_jobs_done = 0
    citation_audit_jobs_done = 0

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
                            max_papers_per_query=max(8, payload.max_papers_per_query),
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
                                max_papers_per_query=max(8, payload.max_papers_per_query),
                            )
                            conn_retry.commit()
                        except sqlite3.DatabaseError:
                            conn_retry.rollback()
                            raise
                        finally:
                            conn_retry.close()
                recompute_insights(user_id, target_pairs={(int(item_id), int(symptom_id))})
                evidence_jobs_done += 1
            elif job["job_type"] == JOB_MODEL_RETRAIN:
                run_training(dataset_source=os.getenv("MODEL_RETRAIN_DATASET_SOURCE", "hybrid"))
                mark_model_retrain_completed()
                model_retrain_jobs_done += 1
            elif job["job_type"] == JOB_CITATION_AUDIT:
                limit = int(job.get("payload", {}).get("limit", 300) or 300)
                delete_missing = bool(job.get("payload", {}).get("delete_missing", True))
                conn = get_connection()
                try:
                    audit_claim_citations(
                        conn,
                        limit=max(1, min(limit, 5000)),
                        delete_missing=delete_missing,
                    )
                    conn.commit()
                except sqlite3.DatabaseError:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
                citation_audit_jobs_done += 1
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
        "model_retrain_jobs_done": model_retrain_jobs_done,
        "citation_audit_jobs_done": citation_audit_jobs_done,
    }

# list insights per user
@app.get("/insights", response_model=list[InsightOut])
def get_insights(user_id: Optional[int] = None, include_suppressed: bool = True, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    return list_insights(user_id=user_id, include_suppressed=include_suppressed)


@app.get("/events/insight_links", response_model=list[EventInsightLinkOut])
def get_event_insight_links(user_id: Optional[int] = None, supported_only: bool = True, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    return list_event_insight_links(user_id=user_id, supported_only=supported_only)


@app.post("/insights/{insight_id}/verify", response_model=InsightVerifyOut)
def verify_insight(insight_id: int, payload: InsightVerifyIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    try:
        result = set_insight_verification(
            user_id=user_id,
            insight_id=insight_id,
            verified=payload.verified,
        )
        maybe_enqueue_model_retrain(trigger_user_id=int(user_id))
        return result
    except ValueError as exc:
        if str(exc) == "insight_not_found":
            raise HTTPException(status_code=404, detail="Insight not found")
        raise


@app.post("/insights/{insight_id}/reject", response_model=InsightVerifyOut)
def reject_insight(insight_id: int, payload: InsightRejectIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
        allow_legacy_explicit=True,
    )
    try:
        result = set_insight_rejection(
            user_id=user_id,
            insight_id=insight_id,
            rejected=payload.rejected,
        )
        maybe_enqueue_model_retrain(trigger_user_id=int(user_id))
        if "verified" not in result:
            result["verified"] = False
        return result
    except ValueError as exc:
        if str(exc) == "insight_not_found":
            raise HTTPException(status_code=404, detail="Insight not found")
        raise
