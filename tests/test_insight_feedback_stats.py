from __future__ import annotations

import unittest

import api.db
from api.main import get_insight_feedback_stats
from tests.db_test_utils import reset_test_database


class InsightFeedbackStatsTests(unittest.TestCase):
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
        self._exec("INSERT INTO users (id, created_at, name) VALUES (1, '2026-01-01T00:00:00Z', 'u1')")
        self._exec("INSERT INTO users (id, created_at, name) VALUES (2, '2026-01-01T00:00:00Z', 'u2')")
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'coffee', 'food')")
        self._exec("INSERT INTO items (id, name, category) VALUES (2, 'milk', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'headache', 'd')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (2, 'bloating', 'd')")

        self._exec(
            """
            INSERT INTO insights (id, user_id, item_id, symptom_id, display_decision_reason, created_at)
            VALUES (1, 1, 1, 1, 'supported', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insights (id, user_id, item_id, symptom_id, display_decision_reason, created_at)
            VALUES (2, 1, 2, 2, 'suppressed_low_overall_confidence', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insights (id, user_id, item_id, symptom_id, display_decision_reason, created_at)
            VALUES (3, 1, 2, 1, NULL, '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insights (id, user_id, item_id, symptom_id, display_decision_reason, created_at)
            VALUES (4, 2, 1, 1, 'supported', '2026-01-01T00:00:00Z')
            """
        )

        self._exec(
            """
            INSERT INTO insight_verifications (user_id, item_id, symptom_id, verified, rejected, created_at, updated_at)
            VALUES (1, 1, 1, 1, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insight_verifications (user_id, item_id, symptom_id, verified, rejected, created_at, updated_at)
            VALUES (1, 2, 2, 0, 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insight_verifications (user_id, item_id, symptom_id, verified, rejected, created_at, updated_at)
            VALUES (1, 2, 1, 0, 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO insight_verifications (user_id, item_id, symptom_id, verified, rejected, created_at, updated_at)
            VALUES (2, 1, 1, 1, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """
        )

    def test_counts_only_user_feedback_for_surfaced_insights(self) -> None:
        result = get_insight_feedback_stats(user_id=1)
        self.assertEqual(result["user_id"], 1)
        self.assertEqual(result["verified_count"], 1)
        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["total_count"], 2)
        self.assertTrue(result["surfaced_only"])

    def test_returns_zero_when_no_feedback(self) -> None:
        self._exec("INSERT INTO users (id, created_at, name) VALUES (3, '2026-01-01T00:00:00Z', 'u3')")
        result = get_insight_feedback_stats(user_id=3)
        self.assertEqual(result["user_id"], 3)
        self.assertEqual(result["verified_count"], 0)
        self.assertEqual(result["rejected_count"], 0)
        self.assertEqual(result["total_count"], 0)
        self.assertTrue(result["surfaced_only"])


if __name__ == "__main__":
    unittest.main()
