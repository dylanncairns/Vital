from __future__ import annotations

import unittest

import api.db
from api.main import (
    RecomputeInsightsIn,
    get_insights,
    recompute_user_insights,
)
from ml.rag import ingest_paper_claim_chunks
from tests.db_test_utils import reset_test_database


class InsightsIntegrationTests(unittest.TestCase):
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

    def test_recompute_then_get_insights_contains_citations_and_evidence(self) -> None:
        recompute_payload = RecomputeInsightsIn(user_id=1, online_enabled=False, max_papers_per_query=1)
        recompute_result = recompute_user_insights(recompute_payload)
        self.assertEqual(recompute_result["status"], "ok")
        self.assertGreaterEqual(recompute_result["insights_written"], 1)

        rows = get_insights(user_id=1, include_suppressed=True)
        self.assertGreaterEqual(len(rows), 1)
        row = rows[0]
        self.assertGreater(row["evidence_strength_score"], 0.0)
        self.assertGreaterEqual(row["evidence_quality_score"], 0.0)
        self.assertLessEqual(row["evidence_quality_score"], 1.0)
        self.assertGreaterEqual(row["overall_confidence_score"], 0.0)
        self.assertLessEqual(row["overall_confidence_score"], 1.0)
        self.assertIn("claim(s) retrieved", row["evidence_summary"])
        self.assertTrue(len(row["citations"]) >= 1)
        self.assertIn(
            row["display_status"],
            {"supported", "insufficient_evidence"},
        )
        self.assertIn(
            row["display_decision_reason"],
            {
                "supported",
                "suppressed_low_evidence_strength",
                "suppressed_low_model_probability",
                "suppressed_low_overall_confidence",
                "suppressed_insufficient_recurrence",
            },
        )


if __name__ == "__main__":
    unittest.main()
