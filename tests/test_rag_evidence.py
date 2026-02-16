from __future__ import annotations

import json
import unittest

import api.db
from ml.rag import (
    _llm_retrieve_evidence_rows,
    aggregate_evidence,
    build_candidate_query,
    enrich_claims_for_candidates,
    ingest_paper_claim_chunks,
    retrieve_claim_evidence,
)
from tests.db_test_utils import reset_test_database


class RagEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_test_database()
        self._seed_basics()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        conn = api.db.get_connection()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _seed_basics(self) -> None:
        self._exec("INSERT INTO users (id, created_at, name) VALUES (1, '2026-01-01T00:00:00Z', 'u')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (1, 'sugar', 'd')")
        self._exec("INSERT INTO ingredients (id, name, description) VALUES (2, 'sles', 'd')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (1, 'acne', 'd')")
        self._exec("INSERT INTO symptoms (id, name, description) VALUES (2, 'headache', 'd')")
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (1, 'Sugar Acne Study', 'https://example.org/a', 'high glycemic load', '2024-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )
        self._exec(
            """
            INSERT INTO papers (id, title, url, abstract, publication_date, source, ingested_at)
            VALUES (2, 'SLES Acne Review', 'https://example.org/b', 'surfactant review', '2023-01-01', 'seed', '2026-01-01T00:00:00Z')
            """
        )

    def test_claim_retrieval_filters_by_ingredient_and_symptom(self) -> None:
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=1,
                item_id=None,
                ingredient_id=1,
                symptom_id=1,
                summary="glycemic load can worsen acne in some populations",
                evidence_polarity_and_strength=1,
                citation_title="Sugar Acne Study",
                citation_url="https://example.org/a",
                source_text="Higher glycemic load and sugar intake linked to acne flare patterns.",
            )
            # wrong symptom
            ingest_paper_claim_chunks(
                conn,
                paper_id=1,
                item_id=None,
                ingredient_id=1,
                symptom_id=2,
                summary="sugar claim for headache",
                evidence_polarity_and_strength=1,
                citation_title="Sugar Acne Study",
                citation_url="https://example.org/a",
                source_text="Sugar and headache text.",
            )
            # wrong ingredient
            ingest_paper_claim_chunks(
                conn,
                paper_id=2,
                item_id=None,
                ingredient_id=2,
                symptom_id=1,
                summary="sles claim for acne",
                evidence_polarity_and_strength=1,
                citation_title="SLES Acne Review",
                citation_url="https://example.org/b",
                source_text="SLES and acne-like irritation text.",
            )
            conn.commit()

            query = build_candidate_query(
                item_name="sugar",
                symptom_name="acne",
                routes={"ingestion"},
                lag_bucket_counts={"6_24h": 2},
            )
            results = retrieve_claim_evidence(
                conn,
                ingredient_ids={1},
                item_id=None,
                symptom_id=1,
                query_text=query,
                top_k=5,
            )
        finally:
            conn.close()

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(int(row["ingredient_id"]) == 1 for row in results))
        self.assertTrue(all(int(row["symptom_id"]) == 1 for row in results))

    def test_claim_retrieval_falls_back_to_item_ingredients_when_no_expansions(self) -> None:
        self._exec("INSERT INTO items (id, name, category) VALUES (10, 'sugar drink', 'food')")
        self._exec("INSERT INTO items_ingredients (item_id, ingredient_id) VALUES (10, 1)")
        conn = api.db.get_connection()
        try:
            ingest_paper_claim_chunks(
                conn,
                paper_id=1,
                item_id=None,
                ingredient_id=1,
                symptom_id=1,
                summary="ingredient-linked sugar acne claim",
                evidence_polarity_and_strength=1,
                citation_title="Sugar Acne Study",
                citation_url="https://example.org/a",
                source_text="Higher glycemic load and sugar intake linked to acne flare patterns.",
            )
            conn.commit()

            query = build_candidate_query(
                item_name="sugar drink",
                symptom_name="acne",
                routes={"ingestion"},
                lag_bucket_counts={"6_24h": 1},
            )
            results = retrieve_claim_evidence(
                conn,
                ingredient_ids=set(),
                item_id=10,
                symptom_id=1,
                query_text=query,
                top_k=5,
            )
        finally:
            conn.close()

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(int(row["symptom_id"]) == 1 for row in results))

    def test_aggregate_evidence_scores_and_citations(self) -> None:
        aggregated = aggregate_evidence(
            [
                {
                    "evidence_polarity_and_strength": 1,
                    "relevance": 0.9,
                    "citation_title": "A",
                    "citation_url": "https://a",
                    "citation_snippet": "snippet a",
                    "summary": "a",
                    "chunk_text": "a",
                },
                {
                    "evidence_polarity_and_strength": -1,
                    "relevance": 0.2,
                    "citation_title": "B",
                    "citation_url": "https://b",
                    "citation_snippet": "snippet b",
                    "summary": "b",
                    "chunk_text": "b",
                },
            ]
        )
        self.assertGreater(aggregated["evidence_score"], 0.0)
        self.assertEqual(len(aggregated["citations"]), 2)
        self.assertIn("claim(s) retrieved", aggregated["evidence_summary"])

        empty = aggregate_evidence([])
        self.assertEqual(empty["evidence_score"], 0.0)
        self.assertEqual(empty["citations"], [])

    def test_enrich_claims_for_candidates_online(self) -> None:
        def fake_llm_retriever(
            *,
            symptom_name: str,
            ingredient_names: list[str],
            item_name: str | None,
            routes: list[str] | None = None,
            lag_bucket_counts: dict[str, int] | None = None,
            max_evidence_rows: int,
        ) -> list[dict]:
            _ = (symptom_name, ingredient_names, item_name, routes, lag_bucket_counts, max_evidence_rows)
            return [
                {
                    "title": "Dietary sugar and acne severity",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/777/",
                    "publication_date": "2025",
                    "source": "Clinical Nutrition",
                    "ingredient_name": "sugar",
                    "symptom_name": "acne",
                    "summary": "Higher sugar exposure was associated with acne severity in this cohort.",
                    "snippet": "Higher sugar exposure was associated with acne severity in this cohort.",
                    "evidence_polarity_and_strength": 1,
                }
            ]

        conn = api.db.get_connection()
        try:
            result = enrich_claims_for_candidates(
                conn,
                candidates=[{"item_id": 1, "symptom_id": 1, "ingredient_ids": {1}}],
                ingredient_name_map={1: "sugar"},
                symptom_name_map={1: "acne"},
                item_name_map={1: "soda"},
                online_enabled=True,
                max_papers_per_query=1,
                llm_retriever=fake_llm_retriever,
            )
            conn.commit()
            claims_count = conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"]
            papers_count = conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]
        finally:
            conn.close()

        self.assertGreaterEqual(result["queries_built"], 1)
        self.assertGreaterEqual(result["papers_added"], 1)
        self.assertGreaterEqual(result["claims_added"], 1)
        self.assertGreaterEqual(papers_count, 3)  # 2 seeded + 1 fetched
        self.assertGreaterEqual(claims_count, 1)

    def test_enrich_claims_item_only_candidate(self) -> None:
        self._exec("INSERT INTO items (id, name, category) VALUES (10, 'bad sleep', 'lifestyle')")

        def fake_llm_retriever(
            *,
            symptom_name: str,
            ingredient_names: list[str],
            item_name: str | None,
            routes: list[str] | None = None,
            lag_bucket_counts: dict[str, int] | None = None,
            max_evidence_rows: int,
        ) -> list[dict]:
            _ = (symptom_name, ingredient_names, item_name, routes, lag_bucket_counts, max_evidence_rows)
            return [
                {
                    "title": "Sleep disruption and headache burden",
                    "url": "https://example.org/sleep-headache",
                    "publication_date": "2022",
                    "source": "Neurology Journal",
                    "item_name": "bad sleep",
                    "ingredient_name": None,
                    "symptom_name": "headache",
                    "summary": "Poor sleep quality was associated with higher headache burden.",
                    "snippet": "Poor sleep quality was associated with higher headache burden.",
                    "evidence_polarity_and_strength": 1,
                }
            ]

        conn = api.db.get_connection()
        try:
            result = enrich_claims_for_candidates(
                conn,
                candidates=[{"item_id": 10, "symptom_id": 2, "ingredient_ids": set()}],
                ingredient_name_map={},
                symptom_name_map={2: "headache"},
                item_name_map={10: "bad sleep"},
                online_enabled=True,
                max_papers_per_query=1,
                llm_retriever=fake_llm_retriever,
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT item_id, ingredient_id, symptom_id
                FROM claims
                WHERE symptom_id = 2
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        self.assertGreaterEqual(result["claims_added"], 1)
        assert row is not None
        self.assertEqual(row["item_id"], 10)
        self.assertIsNone(row["ingredient_id"])
        self.assertEqual(row["symptom_id"], 2)

    def test_enrich_claims_combo_candidate_indexes_both_items(self) -> None:
        self._exec("INSERT INTO items (id, name, category) VALUES (30, 'alcohol', 'food')")
        self._exec("INSERT INTO items (id, name, category) VALUES (31, 'poor sleep', 'lifestyle')")

        def fake_llm_retriever(
            *,
            symptom_name: str,
            ingredient_names: list[str],
            item_name: str | None,
            secondary_item_name: str | None = None,
            routes: list[str] | None = None,
            lag_bucket_counts: dict[str, int] | None = None,
            max_evidence_rows: int,
        ) -> list[dict]:
            _ = (
                symptom_name,
                ingredient_names,
                item_name,
                secondary_item_name,
                routes,
                lag_bucket_counts,
                max_evidence_rows,
            )
            return [
                {
                    "title": "Alcohol and sleep loss jointly increase headache risk",
                    "url": "https://example.org/alcohol-sleep-headache",
                    "publication_date": "2024",
                    "source": "Headache Journal",
                    "item_name": "alcohol",
                    "ingredient_name": None,
                    "symptom_name": "headache",
                    "summary": "Combined alcohol use and poor sleep were linked to more headache episodes.",
                    "snippet": "Combined alcohol use and poor sleep were linked to more headache episodes.",
                    "evidence_polarity_and_strength": 1,
                }
            ]

        conn = api.db.get_connection()
        try:
            result = enrich_claims_for_candidates(
                conn,
                candidates=[{"item_id": 30, "secondary_item_id": 31, "symptom_id": 2, "ingredient_ids": set()}],
                ingredient_name_map={},
                symptom_name_map={2: "headache"},
                item_name_map={30: "alcohol", 31: "poor sleep"},
                online_enabled=True,
                max_papers_per_query=1,
                llm_retriever=fake_llm_retriever,
            )
            conn.commit()
            rows = conn.execute(
                """
                SELECT item_id
                FROM claims
                WHERE symptom_id = 2
                  AND citation_url = 'https://example.org/alcohol-sleep-headache'
                ORDER BY item_id
                """
            ).fetchall()
        finally:
            conn.close()

        self.assertGreaterEqual(result["claims_added"], 2)
        self.assertEqual([int(row["item_id"]) for row in rows], [30, 31])

    def test_llm_retrieval_filters_to_grounded_source_urls(self) -> None:
        class FakeResponse:
            def model_dump(self):
                return {
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {
                                "results": [
                                    {"url": "https://allowed.example/paper", "file_id": "file-1", "chunk_id": "chunk-1"}
                                ]
                            },
                        }
                    ],
                    "output_json": {
                        "answer": "a",
                        "confidence": 0.7,
                        "citations": [
                            {
                                "citation_id": "c1",
                                "title": "Allowed",
                                "authors": ["A"],
                                "year": 2024,
                                "doi": None,
                                "url": "https://allowed.example/paper",
                                "file_id": "file-1",
                            },
                            {
                                "citation_id": "c2",
                                "title": "Ungrounded",
                                "authors": ["B"],
                                "year": 2024,
                                "doi": None,
                                "url": "https://not-allowed.example/paper",
                                "file_id": "file-2",
                            },
                        ],
                        "evidence": [
                            {
                                "claim": "Allowed claim about sugar and acne",
                                "supports": [{"citation_id": "c1", "snippet": "sugar acne signal", "chunk_id": "chunk-1"}],
                            },
                            {"claim": "Ungrounded claim", "supports": [{"citation_id": "c2", "snippet": "s2", "chunk_id": "chunk-2"}]},
                        ],
                    },
                }

        class FakeResponses:
            def create(self, **kwargs):  # noqa: ANN003
                _ = kwargs
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        rows = _llm_retrieve_evidence_rows(
            symptom_name="acne",
            ingredient_names=[],
            item_name="sugar",
            routes=["ingestion"],
            lag_bucket_counts={"6_24h": 1},
            max_evidence_rows=5,
            client_override=FakeClient(),
            vector_store_id_override="vs_123",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://allowed.example/paper")


if __name__ == "__main__":
    unittest.main()
