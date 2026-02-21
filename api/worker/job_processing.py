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
            # evidence retrival - first try vector store search, if no citations then queue web search ingestion
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