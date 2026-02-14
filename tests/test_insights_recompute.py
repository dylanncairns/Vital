from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import api.db
from ml.insights import recompute_insights
from ml.rag import ingest_paper_claim_chunks


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
            SELECT model_probability, evidence_strength_score, evidence_quality_score,
                   penalty_score, final_score, display_decision_reason
            FROM insights
            WHERE user_id = 1 AND item_id = 1 AND symptom_id = 1
            """
        )
        # test created insights
        assert insight is not None
        self.assertGreaterEqual(float(insight["model_probability"] or 0.0), 0.0)
        self.assertLessEqual(float(insight["model_probability"] or 0.0), 1.0)
        self.assertAlmostEqual(insight["evidence_strength_score"], 0.0)
        self.assertAlmostEqual(insight["evidence_quality_score"], 0.0)
        self.assertGreaterEqual(float(insight["penalty_score"] or 0.0), 0.0)
        self.assertGreaterEqual(float(insight["final_score"] or 0.0), 0.0)
        self.assertLessEqual(float(insight["final_score"] or 0.0), 1.0)
        self.assertIn(
            insight["display_decision_reason"],
            {
                "suppressed_no_citations",
                "suppressed_low_evidence_strength",
                "suppressed_low_model_probability",
                "suppressed_low_overall_confidence",
                "suppressed_non_supportive_direction",
                "suppressed_insufficient_recurrence",
            },
        )

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

    def test_recompute_backfills_missing_expansions_for_evidence_retrieval(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (4, '2026-01-01T00:00:00Z', 'u4')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (4, 'sugar drink', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (4, 'acne', 'd')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (4, 'refined sugar', 'd')")
        self._exec("INSERT INTO items_ingredients (item_id, ingredient_id) VALUES (4, 4)")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (4, 'Sugar Acne Study', 'https://example.org/sugar-acne', 'high glycemic load', '2024-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=4,
                item_id=None,
                ingredient_id=4,
                symptom_id=4,
                summary="sugar linked to acne flares",
                evidence_polarity_and_strength=1,
                citation_title="Sugar Acne Study",
                citation_url="https://example.org/sugar-acne",
                source_text="Higher sugar intake linked to acne flare trends.",
            )
            conn.commit()
        finally:
            conn.close()
        # Deliberately do NOT insert exposure_expansions rows; recompute should backfill.
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (40, 4, 4, '2026-01-01T00:00:00Z', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (40, 4, 4, '2026-01-01T08:00:00Z', 3)
            """
        )

        recompute_insights(4)
        insight = self._fetchone(
            """
            SELECT evidence_strength_score, display_decision_reason
            FROM insights
            WHERE user_id = 4 AND item_id = 4 AND symptom_id = 4
            ORDER BY id DESC
            LIMIT 1
            """
        )
        assert insight is not None
        self.assertGreater(float(insight["evidence_strength_score"] or 0.0), 0.0)
        self.assertIn(
            str(insight["display_decision_reason"]),
            {
                "supported",
                "suppressed_low_evidence_strength",
                "suppressed_low_model_probability",
                "suppressed_low_overall_confidence",
                "suppressed_insufficient_recurrence",
                "suppressed_generic_monotone_context",
            },
        )

    def test_recompute_suppresses_single_exposure_even_with_multiple_symptom_hits(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (5, '2026-01-01T00:00:00Z', 'u5')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (5, 'water', 'drink')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (5, 'stomachache', 'd')")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (5, 'Water GI paper', 'https://example.org/water-gi', 'ctx', '2025-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=5,
                item_id=5,
                ingredient_id=None,
                symptom_id=5,
                summary="water links to stomach upset in specific context",
                evidence_polarity_and_strength=1,
                citation_title="Water GI paper",
                citation_url="https://example.org/water-gi",
                source_text="Study mentions water exposure and stomachache outcomes.",
            )
            conn.commit()
        finally:
            conn.close()
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (50, 5, 5, '2026-01-01T12:00:00Z', 'ingestion')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (50, 5, 5, '2026-01-01T15:00:00Z', 3)
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (51, 5, 5, '2026-01-01T18:00:00Z', 2)
            """
        )

        recompute_insights(5)
        insight = self._fetchone(
            """
            SELECT display_decision_reason
            FROM insights
            WHERE user_id = 5 AND item_id = 5 AND symptom_id = 5
            ORDER BY id DESC
            LIMIT 1
            """
        )
        assert insight is not None
        self.assertEqual(str(insight["display_decision_reason"]), "suppressed_insufficient_recurrence")

    def test_recompute_suppresses_generic_work_without_qualifiers(self) -> None:
        self._exec(
            "INSERT INTO users (id, created_at, name) VALUES (6, '2026-01-01T00:00:00Z', 'u6')"
        )
        self._exec("INSERT INTO items (id, name, category) VALUES (6, 'work', 'behavioral')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (6, 'headache', 'd')")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (6, 'Work and headache', 'https://example.org/work-headache', 'ctx', '2025-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=6,
                item_id=6,
                ingredient_id=None,
                symptom_id=6,
                summary="Occupational factors may relate to headache.",
                evidence_polarity_and_strength=1,
                citation_title="Work and headache",
                citation_url="https://example.org/work-headache",
                source_text="Occupational factors may relate to headache.",
            )
            conn.commit()
        finally:
            conn.close()
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route, raw_text)
            VALUES
              (601, 6, 6, '2026-01-01T08:00:00Z', 'behavioral', 'work')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (603, 6, 6, '2026-01-02T12:00:00Z', 3)
            """
        )

        recompute_insights(6)
        insight = self._fetchone(
            """
            SELECT display_decision_reason
            FROM insights
            WHERE user_id = 6 AND item_id = 6 AND symptom_id = 6
            ORDER BY id DESC
            LIMIT 1
            """
        )
        assert insight is not None
        self.assertEqual(str(insight["display_decision_reason"]), "suppressed_generic_monotone_context")


if __name__ == "__main__":
    unittest.main()
