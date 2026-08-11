"""
The Kairos Engine — Unit Normalization Test Suite

Guards the boundary between the engine's percent convention and the risk model's
0.0-1.0 training domain. The bug these cover: telemetry reported 42% battery, the
XGBoost model had only ever seen 0.10-0.60, and the raw value went straight in —
saturating the classifier so 42%, 12% and 5% all scored identically.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.units import normalize_battery_fraction, battery_fraction_to_pct


class TestNormalizeBatteryFraction(unittest.TestCase):

    def test_percent_converts_to_fraction(self):
        self.assertAlmostEqual(normalize_battery_fraction(42), 0.42, places=6)
        self.assertAlmostEqual(normalize_battery_fraction(8.0), 0.08, places=6)

    def test_fraction_passes_through(self):
        self.assertAlmostEqual(normalize_battery_fraction(0.42), 0.42, places=6)
        self.assertAlmostEqual(normalize_battery_fraction(0.08), 0.08, places=6)

    def test_percent_and_fraction_agree(self):
        for pct in [5, 12, 25, 42, 60, 80, 99]:
            self.assertAlmostEqual(
                normalize_battery_fraction(pct),
                normalize_battery_fraction(pct / 100.0),
                places=6,
                msg=f"{pct}% and {pct/100.0} must normalize identically",
            )

    def test_is_idempotent(self):
        for v in [42, 0.42, 100, 0.005, 7]:
            once = normalize_battery_fraction(v)
            self.assertAlmostEqual(normalize_battery_fraction(once), once, places=9)

    def test_full_battery(self):
        self.assertAlmostEqual(normalize_battery_fraction(100), 1.0, places=6)

    def test_empty_battery(self):
        self.assertAlmostEqual(normalize_battery_fraction(0), 0.0, places=6)

    def test_result_always_within_zero_and_one(self):
        for v in [0, 0.5, 1, 42, 100, 150]:
            out = normalize_battery_fraction(v)
            self.assertGreaterEqual(out, 0.0)
            self.assertLessEqual(out, 1.0)

    def test_boundary_one_reads_as_full_to_preserve_idempotency(self):
        """
        1.0 is genuinely ambiguous (1% or 100%?). It resolves to 100% so that
        normalize(normalize(100)) stays 1.0 — re-scaling an already-normalized
        full pack down to 1% is the worse failure. See the module docstring.
        """
        self.assertAlmostEqual(normalize_battery_fraction(1.0), 1.0, places=6)
        self.assertAlmostEqual(normalize_battery_fraction(normalize_battery_fraction(100)), 1.0, places=6)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            normalize_battery_fraction(-5)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            normalize_battery_fraction(float("nan"))

    def test_roundtrip_to_percent(self):
        self.assertAlmostEqual(battery_fraction_to_pct(normalize_battery_fraction(42)), 42.0, places=1)


class TestRiskToolsAcceptBothConventions(unittest.TestCase):
    """The whole point: the same battery level scores the same either way."""

    def test_assess_risk_agrees_across_conventions(self):
        from src.tools.implementations import assess_risk
        for pct in [5, 12, 25, 42, 55]:
            as_pct = assess_risk(pct, 18.0, 3200, "CONTINUE")
            as_frac = assess_risk(pct / 100.0, 18.0, 3200, "CONTINUE")
            self.assertEqual(
                as_pct["crash_probability"], as_frac["crash_probability"],
                f"{pct}% and {pct/100.0} disagree — unit mismatch has returned",
            )
            self.assertEqual(as_pct["risk_level"], as_frac["risk_level"])

    def test_rule_based_fallback_agrees_across_conventions(self):
        from src.tools.implementations import _rule_based_risk
        for pct in [10, 22, 30, 55]:
            self.assertEqual(
                _rule_based_risk(pct, 18.0, 3200, "CONTINUE")["crash_probability"],
                _rule_based_risk(pct / 100.0, 18.0, 3200, "CONTINUE")["crash_probability"],
            )

    def test_rule_based_fallback_is_monotonic_in_battery(self):
        """Less battery must never score as safer, under fixed conditions."""
        from src.tools.implementations import _rule_based_risk
        probs = [
            _rule_based_risk(pct, 18.0, 3200, "CONTINUE")["crash_probability"]
            for pct in [55, 40, 30, 20, 10]
        ]
        for higher, lower in zip(probs, probs[1:]):
            self.assertLessEqual(higher, lower)


if __name__ == "__main__":
    unittest.main()
