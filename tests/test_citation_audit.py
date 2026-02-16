from __future__ import annotations

import unittest

import api.db
from ml.citation_audit import audit_claim_citations
from tests.db_test_utils import reset_test_database


class CitationAuditTests(unittest.TestCase):
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
        self._exec("INSERT INTO items (id, name, category) VALUES (1, 'coffee', 'food')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'headache', 'd')")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES
                (1, 'p1', 'https://ok.example/p1', 'a', '2026-01-01', 'seed', '2026-01-01T00:00:00Z'),
                (2, 'p2', 'https://gone.example/p2', 'a', '2026-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO claims (
                item_id, ingredient_id, symptom_id, paper_id, summary, citation_title, citation_url,
                citation_snippet, chunk_text, chunk_hash, evidence_polarity_and_strength
            )
            VALUES
                (1, NULL, 1, 1, 'ok', 'ok', 'https://ok.example/p1', 's', 's', 'h1', 1),
                (1, NULL, 1, 2, 'gone', 'gone', 'https://gone.example/p2', 's', 's', 'h2', 1)
            """
        )

    def test_audit_deletes_missing_citation_claims(self) -> None:
        conn = api.db.get_connection()
        try:
            result = audit_claim_citations(
                conn,
                limit=10,
                delete_missing=True,
                checker=lambda url: "missing" if "gone.example" in url else "exists",
            )
            conn.commit()
            urls = [row["citation_url"] for row in conn.execute("SELECT citation_url FROM claims").fetchall()]
        finally:
            conn.close()

        self.assertEqual(result["missing_urls"], 1)
        self.assertGreaterEqual(result["deleted_claims"], 1)
        self.assertEqual(urls, ["https://ok.example/p1"])

    def test_audit_does_not_delete_on_probe_error(self) -> None:
        conn = api.db.get_connection()
        try:
            result = audit_claim_citations(
                conn,
                limit=10,
                delete_missing=True,
                checker=lambda _url: "error",
            )
            conn.commit()
            count = int(conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"])
        finally:
            conn.close()

        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["deleted_claims"], 0)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
