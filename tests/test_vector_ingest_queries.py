from __future__ import annotations

import unittest

from ml.vector_ingest import (
    _build_structured_queries_for_candidate,
    _paper_rank_score,
    _tokenize_terms,
)


class VectorIngestQueryTests(unittest.TestCase):
    def test_structured_queries_include_symptom_route_and_onset(self) -> None:
        queries = _build_structured_queries_for_candidate(
            item_name="sugar",
            secondary_item_name=None,
            symptom_name="acne",
            ingredient_names=[],
            routes=["ingestion"],
            lag_bucket_counts={"6_24h": 3, "24_72h": 1},
        )
        joined = "\n".join(queries).lower()
        self.assertTrue(len(queries) >= 4)
        self.assertIn("sugar acne", joined)
        self.assertIn("oral exposure", joined)
        self.assertIn("onset within 24 hours", joined)

    def test_structured_queries_prefer_ingredients_when_available(self) -> None:
        queries = _build_structured_queries_for_candidate(
            item_name="face wash",
            secondary_item_name=None,
            symptom_name="acne",
            ingredient_names=["salicylic acid"],
            routes=["dermal"],
            lag_bucket_counts={"24_72h": 2},
        )
        joined = "\n".join(queries).lower()
        self.assertIn("salicylic acid acne", joined)
        self.assertIn("face wash acne", joined)

    def test_structured_queries_include_combo_phrase(self) -> None:
        queries = _build_structured_queries_for_candidate(
            item_name="alcohol",
            secondary_item_name="poor sleep",
            symptom_name="headache",
            ingredient_names=[],
            routes=["ingestion"],
            lag_bucket_counts={"6_24h": 2},
        )
        joined = "\n".join(queries).lower()
        self.assertIn("alcohol and poor sleep headache", joined)
        self.assertIn("interaction adverse effect", joined)

    def test_tokenize_terms_filters_short_and_stopwords(self) -> None:
        tokens = _tokenize_terms("the oral exposure and acne in humans")
        self.assertIn("oral", tokens)
        self.assertIn("acne", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)

    def test_paper_rank_score_prefers_recent_high_trust_source(self) -> None:
        symptom_tokens = {"headache"}
        exposure_tokens = {"alcohol"}
        recent_pubmed = {
            "title": "Alcohol linked to headache in adults",
            "abstract": "A cohort reported headache outcomes for alcohol exposure.",
            "snippet": "Alcohol exposure associated with headache episodes.",
            "publication_date": "2025-02-01",
            "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "source": "PubMed",
        }
        old_low_trust = {
            "title": "Alcohol and headache",
            "abstract": "General commentary.",
            "snippet": "Anecdotal mention.",
            "publication_date": "2001-01-01",
            "url": "https://example-blog.net/post",
            "source": "blog",
        }
        self.assertGreater(
            _paper_rank_score(
                paper=recent_pubmed,
                symptom_tokens=symptom_tokens,
                exposure_tokens=exposure_tokens,
            ),
            _paper_rank_score(
                paper=old_low_trust,
                symptom_tokens=symptom_tokens,
                exposure_tokens=exposure_tokens,
            ),
        )


if __name__ == "__main__":
    unittest.main()
