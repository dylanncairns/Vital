import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from psycopg import Error as DatabaseError

from api.db import get_connection, initialize_database
from api.repositories.events import list_events
from api.repositories.auth import (
    create_user,
    delete_user_account,
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
from api.event_helpers import (
    enqueue_impacted_recompute_jobs,
    enqueue_recompute_jobs_for_items,
    enqueue_recompute_jobs_for_symptoms,
    event_response,
    insert_event_and_expand,
    normalize_patch_time_value,
)
from api.schemas import (
    AuthLoginIn,
    AuthMePatchIn,
    AuthRegisterIn,
    AuthTokenOut,
    AuthUserOut,
    CitationAuditEnqueueOut,
    CitationAuditIn,
    CitationAuditOut,
    EventIn,
    EventInsightLinkOut,
    EventMutationOut,
    EventOut,
    EventPatchIn,
    InsightOut,
    InsightRejectIn,
    InsightVerifyIn,
    InsightVerifyOut,
    ProcessJobsIn,
    ProcessJobsOut,
    RagSyncIn,
    RagSyncOut,
    RecomputeInsightsIn,
    RecomputeInsightsOut,
    RecurringExposureRuleIn,
    RecurringExposureRuleOut,
    RecurringExposureRulePatchIn,
    TextIngestIn,
    TextIngestOut,
    TimelineEvent,
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
    normalize_event,
    normalize_route,
)
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
logger = logging.getLogger(__name__)


def _maybe_enqueue_model_retrain_safe(*, trigger_user_id: int) -> None:
    try:
        maybe_enqueue_model_retrain(trigger_user_id=trigger_user_id)
    except Exception:
        # Mutations should succeed even if background retrain enqueue fails.
        logger.exception("Model retrain enqueue failed", extra={"trigger_user_id": int(trigger_user_id)})

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
) -> int:
    token = _extract_bearer_token(authorization)
    auth_user = resolve_user_from_token(token) if token else None
    if auth_user:
        auth_user_id = int(auth_user["id"])
        if explicit_user_id is not None and int(explicit_user_id) != auth_user_id:
            raise HTTPException(status_code=403, detail="user_id does not match auth token")
        return auth_user_id
    # Allow direct function invocation in tests only when no auth header is provided.
    if (
        explicit_user_id is not None
        and authorization is None
        and os.getenv("APP_ENV", "").strip().lower() == "test"
    ):
        return int(explicit_user_id)
    raise HTTPException(status_code=401, detail="Authentication required")


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
            "UPDATE users SET name = %s WHERE id = %s",
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


@app.delete("/auth/me")
def auth_me_delete(authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)
    auth_user = resolve_user_from_token(token)
    if auth_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    delete_user_account(user_id=int(auth_user["id"]))
    return {"status": "ok"}


# user submits entry
@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn, authorization: Optional[str] = Header(default=None)):
    payload = payload.model_copy(
        update={
            "user_id": _resolve_request_user_id(
                explicit_user_id=payload.user_id,
                authorization=authorization,
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
        return event_response(-1, payload_dict, status="queued", resolution="pending")

    try:
        created_id = insert_event_and_expand(normalized)
    except DatabaseError as exc:
        # db write failed, queue for retry/review instead of surfacing stack trace
        insert_raw_event_ingest(normalized["user_id"], json.dumps(normalized), "failed", str(exc))
        return event_response(-1, normalized, status="queued", resolution="db_error")

    enqueue_impacted_recompute_jobs(normalized)
    _maybe_enqueue_model_retrain_safe(trigger_user_id=int(normalized["user_id"]))
    return event_response(created_id, normalized)


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
    )
    normalized_type = event_type.strip().lower()
    if normalized_type not in {"exposure", "symptom"}:
        raise HTTPException(status_code=400, detail="event_type must be exposure or symptom")

    conn = get_connection()
    try:
        if normalized_type == "exposure":
            existing = conn.execute(
                "SELECT id, item_id FROM exposure_events WHERE id = %s AND user_id = %s",
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
                updates.append("item_id = %s")
                params.append(int(resolved_item_id))
            if payload.route is not None:
                updates.append("route = %s")
                try:
                    normalized_route = normalize_route(payload.route, strict=True)
                except NormalizationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                params.append(normalized_route)
            if payload.timestamp is not None:
                updates.append("timestamp = %s")
                try:
                    params.append(normalize_patch_time_value(payload.timestamp))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: timestamp")
            if payload.time_range_start is not None:
                updates.append("time_range_start = %s")
                try:
                    params.append(normalize_patch_time_value(payload.time_range_start))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_start")
            if payload.time_range_end is not None:
                updates.append("time_range_end = %s")
                try:
                    params.append(normalize_patch_time_value(payload.time_range_end))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_end")
            if payload.time_confidence is not None:
                updates.append("time_confidence = %s")
                params.append(payload.time_confidence)
            if payload.raw_text is not None:
                updates.append("raw_text = %s")
                params.append(payload.raw_text)
            if updates:
                params.extend([int(event_id), int(user_id)])
                conn.execute(
                    f"UPDATE exposure_events SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                    tuple(params),
                )
                conn.execute("DELETE FROM exposure_expansions WHERE exposure_event_id = %s", (int(event_id),))
                expand_exposure_event(int(event_id), conn=conn)
            conn.commit()
            old_item_id = int(existing["item_id"])
            new_item_id = int(resolved_item_id) if resolved_item_id is not None else old_item_id
            jobs_queued = enqueue_recompute_jobs_for_items(
                user_id=int(user_id),
                item_ids={old_item_id, new_item_id},
                trigger="event_patch_exposure",
            )
        else:
            existing = conn.execute(
                "SELECT id, symptom_id FROM symptom_events WHERE id = %s AND user_id = %s",
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
                updates.append("symptom_id = %s")
                params.append(int(resolved_symptom_id))
            if payload.severity is not None:
                updates.append("severity = %s")
                params.append(int(payload.severity))
            if payload.timestamp is not None:
                updates.append("timestamp = %s")
                try:
                    params.append(normalize_patch_time_value(payload.timestamp))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: timestamp")
            if payload.time_range_start is not None:
                updates.append("time_range_start = %s")
                try:
                    params.append(normalize_patch_time_value(payload.time_range_start))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_start")
            if payload.time_range_end is not None:
                updates.append("time_range_end = %s")
                try:
                    params.append(normalize_patch_time_value(payload.time_range_end))
                except ValueError:
                    raise HTTPException(status_code=400, detail="invalid datetime format: time_range_end")
            if payload.time_confidence is not None:
                updates.append("time_confidence = %s")
                params.append(payload.time_confidence)
            if payload.raw_text is not None:
                updates.append("raw_text = %s")
                params.append(payload.raw_text)
            if updates:
                params.extend([int(event_id), int(user_id)])
                conn.execute(
                    f"UPDATE symptom_events SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                    tuple(params),
                )
            conn.commit()
            old_symptom_id = int(existing["symptom_id"])
            new_symptom_id = int(resolved_symptom_id) if resolved_symptom_id is not None else old_symptom_id
            jobs_queued = enqueue_recompute_jobs_for_symptoms(
                user_id=int(user_id),
                symptom_ids={old_symptom_id, new_symptom_id},
                trigger="event_patch_symptom",
            )

        _maybe_enqueue_model_retrain_safe(trigger_user_id=int(user_id))
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
    )
    normalized_type = event_type.strip().lower()
    if normalized_type not in {"exposure", "symptom"}:
        raise HTTPException(status_code=400, detail="event_type must be exposure or symptom")
    conn = get_connection()
    try:
        if normalized_type == "exposure":
            existing = conn.execute(
                "SELECT item_id FROM exposure_events WHERE id = %s AND user_id = %s",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            item_id = int(existing["item_id"])
            conn.execute("DELETE FROM exposure_expansions WHERE exposure_event_id = %s", (int(event_id),))
            conn.execute("DELETE FROM exposure_events WHERE id = %s AND user_id = %s", (int(event_id), int(user_id)))
            conn.commit()
            jobs_queued = enqueue_recompute_jobs_for_items(
                user_id=int(user_id),
                item_ids={item_id},
                trigger="event_delete_exposure",
            )
        else:
            existing = conn.execute(
                "SELECT symptom_id FROM symptom_events WHERE id = %s AND user_id = %s",
                (int(event_id), int(user_id)),
            ).fetchone()
            if existing is None:
                return {"status": "not_found", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": 0}
            symptom_id = int(existing["symptom_id"])
            conn.execute("DELETE FROM symptom_events WHERE id = %s AND user_id = %s", (int(event_id), int(user_id)))
            conn.commit()
            jobs_queued = enqueue_recompute_jobs_for_symptoms(
                user_id=int(user_id),
                symptom_ids={symptom_id},
                trigger="event_delete_symptom",
            )
        _maybe_enqueue_model_retrain_safe(trigger_user_id=int(user_id))
        return {"status": "ok", "event_id": int(event_id), "event_type": normalized_type, "jobs_queued": int(jobs_queued)}
    finally:
        conn.close()


@app.get("/recurring_exposures", response_model=list[RecurringExposureRuleOut])
def get_recurring_exposures(user_id: Optional[int] = None, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
    )
    return list_recurring_rules(int(user_id))


@app.post("/recurring_exposures", response_model=RecurringExposureRuleOut)
def create_recurring_exposure(payload: RecurringExposureRuleIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
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
    )
    deleted = delete_recurring_rule(user_id=int(user_id), rule_id=int(rule_id))
    return {"status": "ok" if deleted else "not_found", "rule_id": int(rule_id)}


# timeline display
@app.get("/events", response_model=list[TimelineEvent])
def get_events(user_id: Optional[int] = None, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
    )
    inserted_item_ids = materialize_recurring_exposures(user_id=int(user_id))
    if inserted_item_ids:
        enqueue_recompute_jobs_for_items(
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
    except DatabaseError:
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
def audit_citations(payload: CitationAuditIn, authorization: Optional[str] = Header(default=None)):
    _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
    )
    conn = get_connection()
    try:
        result = audit_claim_citations(
            conn,
            limit=payload.limit,
            delete_missing=payload.delete_missing,
        )
        conn.commit()
    except DatabaseError:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"status": "ok", **result}


@app.post("/citations/audit/enqueue", response_model=CitationAuditEnqueueOut)
def enqueue_citation_audit(payload: CitationAuditIn, authorization: Optional[str] = Header(default=None)):
    enqueue_user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
    )
    job_id = enqueue_background_job(
        user_id=enqueue_user_id,
        job_type=JOB_CITATION_AUDIT,
        item_id=None,
        symptom_id=None,
        payload={"limit": int(payload.limit), "delete_missing": bool(payload.delete_missing)},
    )
    return {"status": "ok", "job_id": job_id}


def process_background_jobs_batch(payload: ProcessJobsIn):
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
                        WHERE user_id = %s AND item_id = %s AND symptom_id = %s
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
                    except DatabaseError:
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
                        except DatabaseError:
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
                except DatabaseError:
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


@app.post("/jobs/process", response_model=ProcessJobsOut)
def process_background_jobs(payload: ProcessJobsIn, authorization: Optional[str] = Header(default=None)):
    _resolve_request_user_id(
        explicit_user_id=None,
        authorization=authorization,
    )
    return process_background_jobs_batch(payload)

# list insights per user
@app.get("/insights", response_model=list[InsightOut])
def get_insights(user_id: Optional[int] = None, include_suppressed: bool = True, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
    )
    return list_insights(user_id=user_id, include_suppressed=include_suppressed)


@app.get("/events/insight_links", response_model=list[EventInsightLinkOut])
def get_event_insight_links(user_id: Optional[int] = None, supported_only: bool = True, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=user_id,
        authorization=authorization,
    )
    return list_event_insight_links(user_id=user_id, supported_only=supported_only)


@app.post("/insights/{insight_id}/verify", response_model=InsightVerifyOut)
def verify_insight(insight_id: int, payload: InsightVerifyIn, authorization: Optional[str] = Header(default=None)):
    user_id = _resolve_request_user_id(
        explicit_user_id=payload.user_id,
        authorization=authorization,
    )
    try:
        result = set_insight_verification(
            user_id=user_id,
            insight_id=insight_id,
            verified=payload.verified,
        )
        _maybe_enqueue_model_retrain_safe(trigger_user_id=int(user_id))
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
    )
    try:
        result = set_insight_rejection(
            user_id=user_id,
            insight_id=insight_id,
            rejected=payload.rejected,
        )
        _maybe_enqueue_model_retrain_safe(trigger_user_id=int(user_id))
        if "verified" not in result:
            result["verified"] = False
        return result
    except ValueError as exc:
        if str(exc) == "insight_not_found":
            raise HTTPException(status_code=404, detail="Insight not found")
        raise
