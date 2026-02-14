from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import api.db
import ingestion.ingest_text as ingest_text_mod
from ingestion.ingest_text import ParsedEvent, ingest_text_event


class IngestTextJobsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_db_path = api.db.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        api.db.DB_PATH = Path(self._tmpdir.name) / "test.db"
        api.db.initialize_database()
        self._orig_parse_text_event = ingest_text_mod.parse_text_event
        self._seed()

    def tearDown(self) -> None:
        ingest_text_mod.parse_text_event = self._orig_parse_text_event
        api.db.DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

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
        self._exec("INSERT INTO items (id, name, category) VALUES (2, 'chicken', 'food')")
        self._exec("INSERT INTO items (id, name, category) VALUES (3, 'rice', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'headache', 'd')")

    def test_ingest_text_symptom_queues_jobs_for_existing_exposures(self) -> None:
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route)
            VALUES (1, 1, 1, '2026-01-01T20:00:00Z', 'ingestion')
            """
        )

        ingest_text_mod.parse_text_event = lambda _text: ParsedEvent(
            event_type="symptom",
            timestamp="2026-01-02T08:00:00Z",
            time_range_start=None,
            time_range_end=None,
            time_confidence="exact",
            item_id=None,
            route=None,
            symptom_id=1,
            severity=3,
        )

        result = ingest_text_event(1, "had headache this morning")
        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["event_type"], "symptom")
        self.assertGreaterEqual(int(result.get("jobs_queued", 0)), 1)

    def test_ingest_text_exposure_queues_jobs_for_existing_symptoms(self) -> None:
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity)
            VALUES (1, 1, 1, '2026-01-02T08:00:00Z', 3)
            """
        )

        ingest_text_mod.parse_text_event = lambda _text: ParsedEvent(
            event_type="exposure",
            timestamp="2026-01-01T20:00:00Z",
            time_range_start=None,
            time_range_end=None,
            time_confidence="exact",
            item_id=1,
            route="ingestion",
            symptom_id=None,
            severity=None,
        )

        result = ingest_text_event(1, "drank alcohol last night")
        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["event_type"], "exposure")
        self.assertGreaterEqual(int(result.get("jobs_queued", 0)), 1)

    def test_ingest_text_fans_out_multi_item_exposure_clause(self) -> None:
        result = ingest_text_event(1, "For lunch I had chicken and rice.")
        self.assertEqual(result["status"], "ingested")
        conn = api.db.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT i.name
                FROM exposure_events e
                JOIN items i ON i.id = e.item_id
                WHERE e.user_id = 1
                ORDER BY i.name
                """
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual([row["name"] for row in rows], ["chicken", "rice"])


if __name__ == "__main__":
    unittest.main()
