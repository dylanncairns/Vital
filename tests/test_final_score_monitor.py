from __future__ import annotations

import unittest

from ml.final_score import should_promote_fusion


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


if __name__ == "__main__":
    unittest.main()
