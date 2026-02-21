from __future__ import annotations

import unittest

import api.db
from ml.training.training_data import build_case_control_training_rows
from tests.db_test_utils import reset_test_database


class TrainingDataTests(unittest.TestCase):
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
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'alcohol', 'food')")
        self._exec("INSERT INTO items (id, name, category) VALUES (2, 'water', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'headache', 'd')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (1, 'ethanol', 'd')")
        self._exec("INSERT INTO items_ingredients (item_id, ingredient_id) VALUES (1, 1)")

        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route, time_confidence)
            VALUES (1, 1, 1, '2026-01-03T08:00:00Z', 'ingestion', 'exact')
            """
        )
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route, time_confidence)
            VALUES (2, 1, 2, '2026-01-03T08:00:00Z', 'ingestion', 'approx')
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
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity, time_confidence)
            VALUES (1, 1, 1, '2026-01-10T20:00:00Z', 3, 'exact')
            """
        )

        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (1, 'Alcohol headache study', 'https://example.org/a', 'a', '2024-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO claims (
                id, item_id, ingredient_id, symptom_id, paper_id, summary, evidence_polarity_and_strength
            )
            VALUES (1, NULL, 1, 1, 1, 'ethanol linked to headache', 1)
            """
        )

    def test_build_case_control_rows(self) -> None:
        conn = api.db.get_connection()
        try:
            rows, labels, groups = build_case_control_training_rows(conn, window_hours=24, controls_per_case=1)
        finally:
            conn.close()

        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(len(rows), len(labels))
        self.assertEqual(len(rows), len(groups))
        self.assertIn(1, labels)
        self.assertIn(0, labels)
        self.assertIn("evidence_strength_score", rows[0])
        self.assertIn("time_gap_min_minutes", rows[0])


if __name__ == "__main__":
    unittest.main()
