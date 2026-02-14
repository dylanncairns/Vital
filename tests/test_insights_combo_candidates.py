from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import api.db
from ml.insights import list_insights, recompute_insights
from ml.rag import ingest_paper_claim_chunks


class InsightsComboCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_db_path = api.db.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        api.db.DB_PATH = Path(self._tmpdir.name) / "test.db"
        api.db.initialize_database()
        self._seed()

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

    def _seed(self) -> None:
        self._exec("INSERT INTO users (id, created_at, name) VALUES (1, '2026-01-01T00:00:00Z', 'u')")
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'alcohol', 'food')")
        self._exec("INSERT INTO items (id, name, category) VALUES (2, 'pizza', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'headache', 'd')")
        # two exposures before same symptom within lag window
        self._exec(
            """
            INSERT INTO exposure_events (id, user_id, item_id, timestamp, route, time_confidence)
            VALUES
                (1, 1, 1, '2026-01-01T18:00:00Z', 'ingestion', 'exact'),
                (2, 1, 2, '2026-01-01T19:00:00Z', 'ingestion', 'exact')
            """
        )
        self._exec(
            """
            INSERT INTO symptom_events (id, user_id, symptom_id, timestamp, severity, time_confidence)
            VALUES (1, 1, 1, '2026-01-02T08:00:00Z', 3, 'exact')
            """
        )

    def test_recompute_writes_combo_insight_row(self) -> None:
        recompute_insights(1)
        conn = api.db.get_connection()
        try:
            row = conn.execute(
                """
                SELECT is_combo, combo_key, secondary_item_id
                FROM insights
                WHERE user_id = 1 AND symptom_id = 1 AND is_combo = 1
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        self.assertEqual(int(row["is_combo"]), 1)
        self.assertEqual(str(row["combo_key"]), "1:2")
        self.assertEqual(int(row["secondary_item_id"]), 2)

    def test_list_insights_formats_combo_item_name(self) -> None:
        recompute_insights(1)
        rows = list_insights(user_id=1, include_suppressed=True)
        combo_rows = [row for row in rows if bool(row.get("is_combo"))]
        self.assertGreaterEqual(len(combo_rows), 1)
        self.assertIn(" + ", str(combo_rows[0]["item_name"]))

    def test_combo_requires_evidence_from_both_items(self) -> None:
        # Only alcohol has direct evidence for headache; pizza has none.
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (11, 'Alcohol and headache', 'https://example.org/a', 'd', '2026-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=11,
                item_id=1,
                ingredient_id=None,
                symptom_id=1,
                summary="Alcohol exposure is associated with headache episodes.",
                evidence_polarity_and_strength=1,
                citation_title="Alcohol and headache",
                citation_url="https://example.org/a",
                source_text="Alcohol exposure is associated with headache episodes.",
            )
            conn.commit()
        finally:
            conn.close()

        recompute_insights(1)
        rows = list_insights(user_id=1, include_suppressed=True)
        combo_rows = [row for row in rows if bool(row.get("is_combo"))]
        self.assertGreaterEqual(len(combo_rows), 1)
        # Combo should not be supported without pair-specific evidence.
        self.assertTrue(
            any(row.get("display_decision_reason") == "suppressed_combo_no_pair_evidence" for row in combo_rows)
        )

    def test_combo_requires_pair_specific_citations(self) -> None:
        # Both items have independent headache evidence, but no single citation mentions both.
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES
                (21, 'Alcohol and headache', 'https://example.org/alcohol', 'd', '2026-01-01', 'seed', '2026-01-01T00:00:00Z'),
                (22, 'Pizza and headache', 'https://example.org/pizza', 'd', '2026-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=21,
                item_id=1,
                ingredient_id=None,
                symptom_id=1,
                summary="Alcohol exposure is associated with headache episodes.",
                evidence_polarity_and_strength=1,
                citation_title="Alcohol and headache",
                citation_url="https://example.org/alcohol",
                source_text="Alcohol exposure is associated with headache episodes.",
            )
            ingest_paper_claim_chunks(
                conn,
                paper_id=22,
                item_id=2,
                ingredient_id=None,
                symptom_id=1,
                summary="Pizza exposure is associated with headache episodes.",
                evidence_polarity_and_strength=1,
                citation_title="Pizza and headache",
                citation_url="https://example.org/pizza",
                source_text="Pizza exposure is associated with headache episodes.",
            )
            conn.commit()
        finally:
            conn.close()

        recompute_insights(1)
        rows = list_insights(user_id=1, include_suppressed=True)
        combo_rows = [row for row in rows if bool(row.get("is_combo"))]
        self.assertGreaterEqual(len(combo_rows), 1)
        self.assertTrue(
            any(row.get("display_decision_reason") == "suppressed_combo_no_pair_evidence" for row in combo_rows)
        )


if __name__ == "__main__":
    unittest.main()
