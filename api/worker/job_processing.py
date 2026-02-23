import json
import os
from psycopg import Error as DatabaseError
from api.db import get_connection
from api.worker.jobs import (
    JOB_CITATION_AUDIT,
    JOB_EVIDENCE_ACQUIRE_CANDIDATE,
    JOB_MODEL_RETRAIN,
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
    list_pending_jobs,
    mark_model_retrain_completed,
    mark_job_done,
    mark_job_failed,
)
from api.worker.citation_audit import audit_claim_citations
from api.schemas import ProcessJobsIn
from ml.rag import (
    fetch_ingredient_name_map,
    fetch_item_name_map,
    fetch_symptom_name_map,
    sync_claims_for_candidates,
)
from ml.insights import list_rag_sync_candidates, recompute_insights
from ml.vector_ingest import ingest_sources_for_candidates
from ml.training.training_pipeline import run_training

EVIDENCE_REACQUIRE_DECISION_REASONS = {
    "suppressed_no_citations",
    "suppressed_low_evidence_strength",
    "suppressed_combo_no_pair_evidence",
    "suppressed_combo_unbalanced_evidence",
}


def _insight_row_has_no_evidence(row) -> bool:
    if row is None:
        return True
    try:
        citations_payload = row.get("citations_json") if isinstance(row, dict) else row["citations_json"]
    except Exception:
        citations_payload = None
    citations: list[object] = []
    if isinstance(citations_payload, str) and citations_payload.strip():
        try:
            decoded = json.loads(citations_payload)
            if isinstance(decoded, list):
                citations = decoded
        except json.JSONDecodeError:
            citations = []
    evidence_summary = ""
    try:
        evidence_summary = str((row.get("evidence_summary") if isinstance(row, dict) else row["evidence_summary"]) or "")
    except Exception:
        evidence_summary = ""
    return (not citations) or ("No matching evidence found for this symptom and exposure pattern." in evidence_summary)

# called by main endpoint that worker repeatedly hits
# may need separation of logic for multiple workers if model retrain is too heavy
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
            # check if recompute reveals insufficient evidence for new candidate
            if job["job_type"] == JOB_RECOMPUTE_CANDIDATE:
                if item_id is None or symptom_id is None:
                    raise ValueError("recompute job missing item_id or symptom_id")
                recompute_insights(user_id, target_pairs={(int(item_id), int(symptom_id))})
                recompute_jobs_done += 1

                conn = get_connection()
                try:
                    row = conn.execute(
                        """
                        SELECT evidence_strength_score, display_decision_reason, citations_json, evidence_summary,
                               COALESCE(is_combo, 0) AS is_combo, secondary_item_id
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
                decision_reason = str(row["display_decision_reason"] or "") if row else ""
                try:
                    reacquire_max_strength = float(os.getenv("EVIDENCE_REACQUIRE_MAX_STRENGTH", "0.35"))
                except (TypeError, ValueError):
                    reacquire_max_strength = 0.35
                should_reacquire = (
                    row is None
                    or _insight_row_has_no_evidence(row)
                    or (
                        decision_reason in EVIDENCE_REACQUIRE_DECISION_REASONS
                        and evidence_strength <= max(0.0, min(1.0, reacquire_max_strength))
                    )
                )
                if should_reacquire:
                    reacquire_payload = {"trigger": "insufficient_evidence"}
                    if row is not None:
                        try:
                            if int(row["is_combo"] or 0) == 1:
                                reacquire_payload["is_combo"] = True
                                if row["secondary_item_id"] is not None:
                                    reacquire_payload["secondary_item_id"] = int(row["secondary_item_id"])
                        except Exception:
                            pass
                    enqueue_background_job(
                        user_id=user_id,
                        job_type=JOB_EVIDENCE_ACQUIRE_CANDIDATE,
                        item_id=int(item_id),
                        symptom_id=int(symptom_id),
                        payload=reacquire_payload,
                    )
            # evidence retrival - first try vector store search, if no citations then queue web search ingestion
            elif job["job_type"] == JOB_EVIDENCE_ACQUIRE_CANDIDATE:
                if item_id is None or symptom_id is None:
                    raise ValueError("evidence job missing item_id or symptom_id")
                payload_dict = job.get("payload") or {}
                requested_is_combo = bool(payload_dict.get("is_combo"))
                requested_secondary_item_id = payload_dict.get("secondary_item_id")
                requested_secondary_item_id_int: int | None = None
                if requested_secondary_item_id is not None:
                    try:
                        requested_secondary_item_id_int = int(requested_secondary_item_id)
                    except (TypeError, ValueError):
                        requested_secondary_item_id_int = None
                candidates = [
                    candidate
                    for candidate in list_rag_sync_candidates(user_id)
                    if int(candidate["item_id"]) == int(item_id)
                    and int(candidate["symptom_id"]) == int(symptom_id)
                    and (
                        (
                            requested_is_combo
                            and candidate.get("secondary_item_id") is not None
                            and requested_secondary_item_id_int is not None
                            and int(candidate["secondary_item_id"]) == requested_secondary_item_id_int
                        )
                        or (
                            not requested_is_combo
                            and candidate.get("secondary_item_id") is None
                        )
                    )
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
                    # web search if vector store has no claims regarding a candidate linkage
                    if int(sync_result.get("claims_added", 0)) == 0:
                        ingest_sources_for_candidates(
                            candidates=candidates,
                            vector_store_id=os.getenv("RAG_VECTOR_STORE_ID"),
                            max_queries=6,
                            max_papers_per_query=max(8, payload.max_papers_per_query),
                        )
                        conn_retry = get_connection()
                        try:
                            sync_retry_result = sync_claims_for_candidates(
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
                    else:
                        sync_retry_result = None
                recompute_insights(user_id, target_pairs={(int(item_id), int(symptom_id))})
                conn_check = get_connection()
                try:
                    if requested_is_combo and requested_secondary_item_id_int is not None:
                        latest_insight = conn_check.execute(
                            """
                            SELECT citations_json, evidence_summary
                            FROM insights
                            WHERE user_id = %s
                              AND item_id = %s
                              AND secondary_item_id = %s
                              AND symptom_id = %s
                              AND COALESCE(is_combo, 0) = 1
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (
                                user_id,
                                int(item_id),
                                requested_secondary_item_id_int,
                                int(symptom_id),
                            ),
                        ).fetchone()
                    else:
                        latest_insight = conn_check.execute(
                            """
                            SELECT citations_json, evidence_summary
                            FROM insights
                            WHERE user_id = %s
                              AND item_id = %s
                              AND symptom_id = %s
                              AND COALESCE(is_combo, 0) = 0
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (user_id, int(item_id), int(symptom_id)),
                        ).fetchone()
                finally:
                    conn_check.close()
                if _insight_row_has_no_evidence(latest_insight):
                    claims_added_total = int(sync_result.get("claims_added", 0) or 0)
                    if isinstance(sync_retry_result, dict):
                        claims_added_total += int(sync_retry_result.get("claims_added", 0) or 0)
                    initial_rows = int(sync_result.get("retrieval_stage_rows", 0) or 0)
                    initial_rejected = int(sync_result.get("rows_rejected_quality", 0) or 0)
                    initial_dupes = int(sync_result.get("duplicate_claim_rows_skipped", 0) or 0)
                    initial_no_rows = int(sync_result.get("candidates_without_rows", 0) or 0)
                    retry_rows = 0
                    retry_rejected = 0
                    retry_dupes = 0
                    retry_no_rows = 0
                    if isinstance(sync_retry_result, dict):
                        retry_rows = int(sync_retry_result.get("retrieval_stage_rows", 0) or 0)
                        retry_rejected = int(sync_retry_result.get("rows_rejected_quality", 0) or 0)
                        retry_dupes = int(sync_retry_result.get("duplicate_claim_rows_skipped", 0) or 0)
                        retry_no_rows = int(sync_retry_result.get("candidates_without_rows", 0) or 0)
                    is_combo_job = bool(payload_dict.get("is_combo"))
                    raise RuntimeError(
                        "no evidence acquired for candidate after retrieval+discovery "
                        f"(is_combo={int(is_combo_job)}, claims_added={claims_added_total}, "
                        f"initial_rows={initial_rows}, initial_rejected={initial_rejected}, "
                        f"initial_dupes={initial_dupes}, initial_no_rows={initial_no_rows}, "
                        f"retry_rows={retry_rows}, retry_rejected={retry_rejected}, "
                        f"retry_dupes={retry_dupes}, retry_no_rows={retry_no_rows})"
                    )
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
