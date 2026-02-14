from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml.final_score import (
    load_fusion_monitor,
    load_score_guardrails,
    save_fusion_monitor,
    should_promote_fusion,
)


class FinalScoreMonitorTests(unittest.TestCase):
    def test_promote_when_no_previous_and_metrics_pass(self) -> None:
        candidate = {
            "auc": 0.71,
            "average_precision": 0.68,
            "brier": 0.19,
            "high_conf_ratio": 0.24,
        }
        promoted, reason = should_promote_fusion(candidate_metrics=candidate, previous_monitor=None)
        self.assertTrue(promoted)
        self.assertEqual(reason, "promote_no_previous_baseline")

    def test_block_on_metric_floor(self) -> None:
        candidate = {
            "auc": 0.54,
            "average_precision": 0.68,
            "brier": 0.19,
            "high_conf_ratio": 0.24,
        }
        promoted, reason = should_promote_fusion(candidate_metrics=candidate, previous_monitor=None)
        self.assertFalse(promoted)
        self.assertEqual(reason, "guardrail_fail_auc_floor")

    def test_block_on_regression_vs_previous(self) -> None:
        previous = {
            "promoted": True,
            "metrics": {
                "auc": 0.74,
                "average_precision": 0.70,
                "brier": 0.17,
                "high_conf_ratio": 0.20,
            },
        }
        candidate = {
            "auc": 0.70,
            "average_precision": 0.69,
            "brier": 0.17,
            "high_conf_ratio": 0.21,
        }
        promoted, reason = should_promote_fusion(candidate_metrics=candidate, previous_monitor=previous)
        self.assertFalse(promoted)
        self.assertEqual(reason, "guardrail_fail_auc_regression")

    def test_save_monitor_writes_history_versions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "fusion.pkl"
            calibrator = root / "cal.pkl"
            thresholds = root / "thresholds.json"
            guardrails = root / "guardrails.json"
            monitor_path = root / "monitor.json"
            model.write_bytes(b"model-v1")
            calibrator.write_bytes(b"cal-v1")
            thresholds.write_text("{}")
            guardrails.write_text("{}")

            save_fusion_monitor(
                metrics={"auc": 0.7, "average_precision": 0.6, "brier": 0.2, "high_conf_ratio": 0.2},
                promoted=True,
                reason="ok",
                dataset_source="hybrid",
                model_path=model,
                calibrator_path=calibrator,
                thresholds_path=thresholds,
                guardrails_path=guardrails,
                path=monitor_path,
            )
            save_fusion_monitor(
                metrics={"auc": 0.71, "average_precision": 0.61, "brier": 0.19, "high_conf_ratio": 0.2},
                promoted=True,
                reason="ok2",
                dataset_source="hybrid",
                model_path=model,
                calibrator_path=calibrator,
                thresholds_path=thresholds,
                guardrails_path=guardrails,
                path=monitor_path,
            )

            payload = load_fusion_monitor(path=monitor_path)
            self.assertIsInstance(payload, dict)
            assert payload is not None
            self.assertIn("versions", payload)
            self.assertIn("history", payload)
            self.assertGreaterEqual(len(payload["history"]), 2)
            self.assertTrue(payload["versions"]["fusion_model"]["exists"])

    def test_load_score_guardrails_defaults_when_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "not_here.json"
            guardrails = load_score_guardrails(path=missing)
            self.assertGreater(guardrails["runtime_min_window"], 0.0)
            self.assertGreater(guardrails["runtime_max_high_conf_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
