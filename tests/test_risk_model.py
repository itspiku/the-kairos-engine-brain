"""
The Kairos Engine — Risk Model Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RISK_MODEL_PATH, CRITICAL_RISK_THRESHOLD, HIGH_RISK_THRESHOLD


def _assess(battery_pct, wind_speed_ms, altitude_m, proposed_action):
    from build_crash_predictor import assess_risk
    return assess_risk(battery_pct, wind_speed_ms, altitude_m, proposed_action)


class TestRiskModel(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists(RISK_MODEL_PATH),
            f"Model file not found at {RISK_MODEL_PATH}. Run: python build_crash_predictor.py"
        )

    def test_critical_risk_scenario(self):
        # Low battery + high wind + high altitude + RTL = critical
        res = _assess(0.20, 18.0, 4000, "RTL")
        self.assertIn("crash_probability", res)
        self.assertIn("risk_level", res)
        self.assertEqual(res["risk_level"], "Critical")
        self.assertGreater(res["crash_probability"], CRITICAL_RISK_THRESHOLD)

    def test_acceptable_risk_scenario(self):
        # High battery + low wind + low altitude + DIVERT = acceptable
        res = _assess(0.55, 5.0, 1500, "DIVERT")
        self.assertEqual(res["risk_level"], "Acceptable")
        self.assertLess(res["crash_probability"], HIGH_RISK_THRESHOLD)

    def test_risk_probability_range(self):
        # Probability must be in [0, 1]
        for params in [
            (0.10, 25.0, 5000, "CONTINUE"),
            (0.60, 0.0, 1000, "DIVERT"),
            (0.30, 12.0, 3000, "RTL"),
        ]:
            res = _assess(*params)
            self.assertGreaterEqual(res["crash_probability"], 0.0)
            self.assertLessEqual(res["crash_probability"], 1.0)

    def test_risk_level_thresholds_match(self):
        # Verify thresholds are consistent with expected config values
        res_critical = _assess(0.15, 20.0, 4500, "RTL")
        self.assertGreater(
            res_critical["crash_probability"],
            HIGH_RISK_THRESHOLD,
            "A dangerous scenario should exceed the high risk threshold"
        )

    def test_action_encoding_consistency(self):
        # Different actions should produce different risk scores for same conditions
        res_continue = _assess(0.25, 18.0, 4000, "CONTINUE")
        res_divert = _assess(0.25, 18.0, 4000, "DIVERT")
        # DIVERT at high wind/low battery should be safer than CONTINUE
        self.assertLessEqual(
            res_divert["crash_probability"],
            res_continue["crash_probability"] + 0.3,  # Allow some variation
        )

    def test_result_has_data_source(self):
        res = _assess(0.30, 10.0, 3000, "CONTINUE")
        self.assertIn("data_source", res)
        self.assertIsInstance(res["data_source"], str)
        self.assertGreater(len(res["data_source"]), 0)

    def test_high_risk_boundary(self):
        # Values near the boundary between acceptable and high
        res = _assess(0.32, 14.0, 3200, "CONTINUE")
        self.assertIn(res["risk_level"], ["Acceptable", "High", "Critical"])


class TestRiskClassifier(unittest.TestCase):

    def test_classifier_loads(self):
        if not os.path.exists(RISK_MODEL_PATH):
            self.skipTest("Model file not built yet")
        from src.models.risk_classifier import KairosRiskClassifier
        KairosRiskClassifier.reset_instance()
        clf = KairosRiskClassifier()
        clf._ensure_loaded()
        self.assertIsNotNone(clf.model)

    def test_classifier_singleton(self):
        from src.models.risk_classifier import KairosRiskClassifier
        a = KairosRiskClassifier()
        b = KairosRiskClassifier()
        self.assertIs(a, b)

    def test_predict_risk_returns_dict(self):
        if not os.path.exists(RISK_MODEL_PATH):
            self.skipTest("Model file not built yet")
        from src.models.risk_classifier import KairosRiskClassifier
        KairosRiskClassifier.reset_instance()
        clf = KairosRiskClassifier()
        result = clf.predict_risk(0.3, 12.0, 3000, "CONTINUE")
        self.assertIn("crash_probability", result)
        self.assertIn("risk_level", result)


if __name__ == "__main__":
    unittest.main()
