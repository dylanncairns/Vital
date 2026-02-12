from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import api.db
from ml.insights import recompute_insights


class InsightsRecomputeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_db_path = api.db.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        api.db.DB_PATH = Path(self._tmpdir.name) / "test.db"
        api.db.initialize_database()

    def tearDown(self) -> None:
        api.db.DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        conn = api.db.get_connection()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        conn = api.db.get_connection()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def test_recompute_derives_expected_feature_values(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (1, '2026-01-01T00:00:00Z', 'u')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'item', 'cat')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'sym', 'd')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (1, 'ing', 'd')")

        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (1, 1, 1, '2026-01-01T00:00:00Z', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (2, 1, 1, '2026-01-03T00:00:00Z', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO exposure_expansions (exposure_event_id, ingredient_id)
            VALUES (2, 1)
            """
        )

        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (1, 1, 1, '2026-01-01T03:00:00Z', 2)
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (2, 1, 1, '2026-01-04T00:00:00Z', 4)
            """
        )
        # Outside max 7d lag window from any exposure so should be ignored
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (3, 1, 1, '2026-01-12T00:00:00Z', 5)
            """
        )

        result = recompute_insights(1)
        self.assertEqual(
            result,
            {
                "candidates_considered": 1,
                "pairs_evaluated": 3,
                "insights_written": 1,
            },
        )

        features = self._fetchone(
            """
            SELECT time_gap_min_minutes, time_gap_avg_minutes, cooccurrence_count,
                   cooccurrence_unique_symptom_count, pair_density,
                   exposure_count_7d, symptom_count_7d, severity_avg_after
            FROM derived_features
            WHERE user_id = 1 AND item_id = 1 AND symptom_id = 1
            """
        )
        # test features created with this input
        assert features is not None
        self.assertAlmostEqual(features["time_gap_min_minutes"], 180.0)
        self.assertAlmostEqual(features["time_gap_avg_minutes"], 1980.0)
        self.assertEqual(features["cooccurrence_count"], 3)
        self.assertEqual(features["cooccurrence_unique_symptom_count"], 2)
        self.assertAlmostEqual(features["pair_density"], 1.5)
        self.assertEqual(features["exposure_count_7d"], 2)
        self.assertEqual(features["symptom_count_7d"], 2)
        self.assertAlmostEqual(features["severity_avg_after"], 10.0 / 3.0)

        insight = self._fetchone(
            """
            SELECT model_probability, evidence_strength_score, final_score, display_decision_reason
            FROM insights
            WHERE user_id = 1 AND item_id = 1 AND symptom_id = 1
            """
        )
        # test created insights
        assert insight is not None
        self.assertAlmostEqual(insight["model_probability"], 0.0)
        self.assertAlmostEqual(insight["evidence_strength_score"], 0.1)
        self.assertAlmostEqual(insight["final_score"], 0.05)
        self.assertEqual(insight["display_decision_reason"], "suppressed_pending_rag_and_model")

        retrieval_count = self._fetchone(
            "SELECT COUNT(*) AS count FROM retrieval_runs WHERE user_id = 1 AND item_id = 1 AND symptom_id = 1"
        )
        assert retrieval_count is not None
        self.assertEqual(retrieval_count["count"], 1)

        ingredient_features = self._fetchone(
            """
            SELECT time_gap_min_minutes, time_gap_avg_minutes, cooccurrence_count,
                   cooccurrence_unique_symptom_count, pair_density,
                   exposure_count_7d, symptom_count_7d, severity_avg_after
            FROM derived_features_ingredients
            WHERE user_id = 1 AND ingredient_id = 1 AND symptom_id = 1
            """
        )
        assert ingredient_features is not None
        self.assertAlmostEqual(ingredient_features["time_gap_min_minutes"], 1440.0)
        self.assertAlmostEqual(ingredient_features["time_gap_avg_minutes"], 1440.0)
        self.assertEqual(ingredient_features["cooccurrence_count"], 1)
        self.assertEqual(ingredient_features["cooccurrence_unique_symptom_count"], 1)
        self.assertAlmostEqual(ingredient_features["pair_density"], 1.0)
        self.assertEqual(ingredient_features["exposure_count_7d"], 1)
        self.assertEqual(ingredient_features["symptom_count_7d"], 2)
        self.assertAlmostEqual(ingredient_features["severity_avg_after"], 4.0)

    def test_recompute_ignores_invalid_timestamps_in_events(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (2, '2026-01-01T00:00:00Z', 'u2')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (2, 'item2', 'cat')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (2, 'sym2', 'd')")

        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (10, 2, 2, 'not-a-timestamp', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (10, 2, 2, '2026-01-02T00:00:00Z', 3)
            """
        )

        result = recompute_insights(2)
        self.assertEqual(
            result,
            {
                "candidates_considered": 0,
                "pairs_evaluated": 0,
                "insights_written": 0,
            },
        )

    def test_recompute_item_only_when_no_expansion_rows(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (3, '2026-01-01T00:00:00Z', 'u3')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (3, 'sleep deprivation', 'lifestyle')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (3, 'fatigue', 'd')")

        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (20, 3, 3, '2026-01-01T00:00:00Z', 'unknown')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (20, 3, 3, '2026-01-01T02:00:00Z', 3)
            """
        )

        result = recompute_insights(3)
        self.assertEqual(
            result,
            {
                "candidates_considered": 1,
                "pairs_evaluated": 1,
                "insights_written": 1,
            },
        )

        item_feature_count = self._fetchone(
            """
            SELECT COUNT(*) AS count
            FROM derived_features
            WHERE user_id = 3 AND item_id = 3 AND symptom_id = 3
            """
        )
        assert item_feature_count is not None
        self.assertEqual(item_feature_count["count"], 1)

        ingredient_feature_count = self._fetchone(
            """
            SELECT COUNT(*) AS count
            FROM derived_features_ingredients
            WHERE user_id = 3
            """
        )
        assert ingredient_feature_count is not None
        self.assertEqual(ingredient_feature_count["count"], 0)


if __name__ == "__main__":
    unittest.main()
