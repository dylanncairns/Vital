from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from ml.evaluator import (
    FEATURE_ORDER,
    apply_probability_calibrator,
    build_training_rows_from_curated_catalog,
    build_feature_vector,
    combine_scores,
    compute_evidence_quality,
    compute_penalty_score,
    fit_probability_calibrator,
    get_decision_thresholds,
    predict_model_probability,
    save_decision_thresholds,
    save_probability_calibrator,
    save_xgboost_artifact,
    train_xgboost_model,
    tune_model_threshold,
)


class EvaluatorTests(unittest.TestCase):
    def test_evidence_quality_and_penalty_ranges(self) -> None:
        evidence = {
            "evidence_strength_score": 0.8,
            "avg_relevance": 0.7,
            "citations": [
                {"evidence_polarity_and_strength": 1},
                {"evidence_polarity_and_strength": 1},
                {"evidence_polarity_and_strength": -1},
            ],
        }
        quality = compute_evidence_quality(evidence)
        self.assertGreaterEqual(quality["score"], 0.0)
        self.assertLessEqual(quality["score"], 1.0)
        self.assertEqual(quality["citation_count"], 3.0)

        penalty = compute_penalty_score(
            {
                "cooccurrence_count": 1.0,
                "exposure_count_7d": 10.0,
                "pair_density": 1.5,
                "time_confidence_score": 0.5,
                "contradict_ratio": quality["contradict_ratio"],
            }
        )
        self.assertGreaterEqual(penalty, 0.0)
        self.assertLessEqual(penalty, 0.75)

        overall = combine_scores(model_probability=0.6, evidence_quality=quality["score"], penalty_score=penalty)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)

    def test_train_and_predict_model(self) -> None:
        x = []
        y = []
        for i in range(24):
            row = [0.0 for _ in FEATURE_ORDER]
            # index 12 is evidence_score_signed, index 13 citation_count
            row[12] = 0.9 if i % 2 == 0 else -0.7
            row[13] = 3.0 if i % 2 == 0 else 0.0
            row[2] = 3.0 if i % 2 == 0 else 1.0
            x.append(row)
            y.append(1 if i % 2 == 0 else 0)

        model = train_xgboost_model(x, y, rounds=12, learning_rate=0.2)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.xgb.json"
            save_xgboost_artifact(model, path=model_path)
            positive_prob = predict_model_probability(
                {name: x[0][i] for i, name in enumerate(FEATURE_ORDER)},
                model_path=model_path,
            )
            negative_prob = predict_model_probability(
                {name: x[1][i] for i, name in enumerate(FEATURE_ORDER)},
                model_path=model_path,
            )
        self.assertGreaterEqual(positive_prob, negative_prob)

    def test_predict_uses_saved_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.xgb.json"
            x = [[0.0 for _ in FEATURE_ORDER] for _ in range(10)]
            y = [1 if i < 5 else 0 for i in range(10)]
            model = train_xgboost_model(x, y, rounds=5, learning_rate=0.2)
            save_xgboost_artifact(model, path=path)

            feature_map = {name: 0.0 for name in FEATURE_ORDER}
            vector = build_feature_vector(feature_map)
            self.assertEqual(len(vector), len(FEATURE_ORDER))
            prob = predict_model_probability(feature_map, model_path=path)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_curated_catalog_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curated.json"
            path.write_text(
                json.dumps(
                    [
                        {"exposure": "alcohol", "symptom": "headache", "label": 1, "strength": 0.9},
                        {"exposure": "water", "symptom": "hangover", "label": 0, "strength": 0.05},
                    ]
                )
            )
            x, y = build_training_rows_from_curated_catalog(path)
            self.assertEqual(len(x), 2)
            self.assertEqual(len(y), 2)
            self.assertEqual(y, [1, 0])
            self.assertEqual(len(x[0]), len(FEATURE_ORDER))

    def test_calibrator_and_thresholds_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator_path = Path(tmpdir) / "cal.pkl"
            thresholds_path = Path(tmpdir) / "thresholds.json"
            calibrator = fit_probability_calibrator([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
            save_probability_calibrator(calibrator, path=calibrator_path)
            self.assertTrue(calibrator_path.exists())
            calibrated = apply_probability_calibrator(0.85, calibrator_path=calibrator_path)
            self.assertGreaterEqual(calibrated, 0.0)
            self.assertLessEqual(calibrated, 1.0)

            threshold = tune_model_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], target_precision=0.75)
            self.assertGreaterEqual(threshold, 0.05)
            self.assertLessEqual(threshold, 0.95)
            save_decision_thresholds(
                min_evidence_strength=0.2,
                min_model_probability=threshold,
                min_overall_confidence=0.5,
                target_precision=0.75,
                source="test",
                path=thresholds_path,
            )
            loaded = get_decision_thresholds(path=thresholds_path)
            self.assertAlmostEqual(loaded["min_evidence_strength"], 0.2)
            self.assertAlmostEqual(loaded["min_overall_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
