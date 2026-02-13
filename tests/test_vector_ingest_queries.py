from __future__ import annotations

import unittest

from ml.vector_ingest import _build_structured_queries_for_candidate


class VectorIngestQueryTests(unittest.TestCase):
    def test_structured_queries_include_symptom_route_and_onset(self) -> None:
        queries = _build_structured_queries_for_candidate(
            item_name="sugar",
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
            symptom_name="acne",
            ingredient_names=["salicylic acid"],
            routes=["dermal"],
            lag_bucket_counts={"24_72h": 2},
        )
        joined = "\n".join(queries).lower()
        self.assertIn("salicylic acid acne", joined)
        self.assertNotIn("face wash acne", joined)


if __name__ == "__main__":
    unittest.main()
