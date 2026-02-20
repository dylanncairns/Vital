from __future__ import annotations

import unittest
from unittest.mock import patch

import api.db
from api.main import ProcessJobsIn, process_background_jobs_batch
from api.repositories.jobs import (
    DEFAULT_MAX_FAILED_ATTEMPTS,
    JOB_CITATION_AUDIT,
    JOB_EVIDENCE_ACQUIRE_CANDIDATE,
    JOB_RECOMPUTE_CANDIDATE,
    count_jobs,
    enqueue_background_job,
    list_pending_jobs,
)
from ml.rag import ingest_paper_claim_chunks
from tests.db_test_utils import reset_test_database


class BackgroundJobsTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_test_database()
        self._seed()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        conn = api.db.get_connection()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _seed(self) -> None:
        self._exec("INSERT INTO users (id, created_at, name) VALUES (1, '2026-01-01T00:00:00Z', 'u')")
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'sugar drink', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'acne', 'd')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (1, 'refined sugar', 'd')")
        self._exec("INSERT INTO items_ingredients (item_id, ingredient_id) VALUES (1, 1)")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (1, 'Sugar Acne Study', 'https://example.org/a', 'high glycemic load', '2024-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (1, 1, 1, '2026-01-01T08:00:00Z', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO exposure_expansions (id, exposure_event_id, ingredient_id)
            VALUES (1, 1, 1)
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (1, 1, 1, '2026-01-01T18:00:00Z', 3)
            """
        )

        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=1,
                item_id=None,
                ingredient_id=1,
                symptom_id=1,
                summary="Sugar exposure is associated with acne flare trends in some cohorts.",
                evidence_polarity_and_strength=1,
                citation_title="Sugar Acne Study",
                citation_url="https://example.org/a",
                source_text="Higher sugar intake and glycemic load were associated with acne flare trends.",
            )
            conn.commit()
        finally:
            conn.close()

    def test_process_recompute_candidate_job(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_RECOMPUTE_CANDIDATE,
            item_id=1,
            symptom_id=1,
            payload={"trigger": "test"},
        )
        self.assertIsNotNone(created)

        result = process_background_jobs_batch(ProcessJobsIn(limit=10, max_papers_per_query=1))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["jobs_claimed"], 1)
        self.assertEqual(result["jobs_done"], 1)
        self.assertEqual(result["jobs_failed"], 0)
        self.assertEqual(result["recompute_jobs_done"], 1)

        conn = api.db.get_connection()
        try:
            row = conn.execute(
                """
                SELECT evidence_strength_score
                FROM insights
                WHERE user_id = 1 AND item_id = 1 AND symptom_id = 1
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        self.assertGreater(float(row["evidence_strength_score"] or 0.0), 0.0)
        self.assertEqual(count_jobs(user_id=1, status="pending"), 0)

    def test_evidence_job_fallback_acquires_sources_when_initial_sync_empty(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_EVIDENCE_ACQUIRE_CANDIDATE,
            item_id=1,
            symptom_id=1,
            payload={"trigger": "test"},
        )
        self.assertIsNotNone(created)

        with (
            patch("api.main.list_rag_sync_candidates", return_value=[{"item_id": 1, "symptom_id": 1, "ingredient_ids": set(), "routes": ["ingestion"], "lag_bucket_counts": {"6_24h": 1}}]),
            patch("api.main.sync_claims_for_candidates", side_effect=[{"queries_built": 1, "papers_added": 0, "claims_added": 0}, {"queries_built": 1, "papers_added": 1, "claims_added": 1}]) as sync_mock,
            patch("api.main.ingest_sources_for_candidates", return_value={"uploaded_count": 1, "source_files": ["x.txt"]}) as ingest_mock,
            patch("api.main.recompute_insights", return_value={"candidates_considered": 1, "pairs_evaluated": 1, "insights_written": 1}),
        ):
            result = process_background_jobs_batch(ProcessJobsIn(limit=10, max_papers_per_query=1))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["jobs_done"], 1)
        self.assertEqual(result["jobs_failed"], 0)
        self.assertEqual(result["evidence_jobs_done"], 1)
        self.assertEqual(sync_mock.call_count, 2)
        ingest_mock.assert_called_once()

    def test_list_pending_jobs_requeues_stale_running(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_RECOMPUTE_CANDIDATE,
            item_id=1,
            symptom_id=1,
            payload={"trigger": "test"},
        )
        self.assertIsNotNone(created)
        assert created is not None

        conn = api.db.get_connection()
        try:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'running',
                    updated_at = '2000-01-01T00:00:00+00:00'
                WHERE id = %s
                """,
                (created,),
            )
            conn.commit()
        finally:
            conn.close()

        claimed = list_pending_jobs(limit=10)
        ids = {row["id"] for row in claimed}
        self.assertIn(created, ids)

    def test_list_pending_jobs_auto_retries_failed_when_backoff_elapsed(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_RECOMPUTE_CANDIDATE,
            item_id=1,
            symptom_id=1,
            payload={"trigger": "test"},
        )
        self.assertIsNotNone(created)
        assert created is not None

        conn = api.db.get_connection()
        try:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'failed',
                    attempts = 1,
                    updated_at = '2000-01-01T00:00:00+00:00'
                WHERE id = %s
                """,
                (created,),
            )
            conn.commit()
        finally:
            conn.close()

        claimed = list_pending_jobs(limit=10)
        ids = {row["id"] for row in claimed}
        self.assertIn(created, ids)

    def test_list_pending_jobs_does_not_retry_failed_over_max_attempts(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_RECOMPUTE_CANDIDATE,
            item_id=1,
            symptom_id=1,
            payload={"trigger": "test"},
        )
        self.assertIsNotNone(created)
        assert created is not None

        conn = api.db.get_connection()
        try:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'failed',
                    attempts = %s,
                    updated_at = '2000-01-01T00:00:00+00:00'
                WHERE id = %s
                """,
                (DEFAULT_MAX_FAILED_ATTEMPTS, created),
            )
            conn.commit()
        finally:
            conn.close()

        claimed = list_pending_jobs(limit=10)
        ids = {row["id"] for row in claimed}
        self.assertNotIn(created, ids)

    def test_process_citation_audit_job(self) -> None:
        created = enqueue_background_job(
            user_id=1,
            job_type=JOB_CITATION_AUDIT,
            item_id=None,
            symptom_id=None,
            payload={"limit": 50, "delete_missing": True},
        )
        self.assertIsNotNone(created)

        with patch(
            "api.main.audit_claim_citations",
            return_value={
                "scanned_urls": 3,
                "missing_urls": 1,
                "deleted_claims": 2,
                "deleted_papers": 1,
                "errors": 0,
            },
        ) as audit_mock:
            result = process_background_jobs_batch(ProcessJobsIn(limit=10, max_papers_per_query=1))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["jobs_done"], 1)
        self.assertEqual(result["jobs_failed"], 0)
        self.assertEqual(result["citation_audit_jobs_done"], 1)
        audit_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
