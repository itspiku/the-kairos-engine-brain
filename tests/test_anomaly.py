"""
The Kairos Engine — Anomaly Detector Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telemetry.anomaly_detector import KairosAnomalyDetector, AnomalySeverity
from src.core.types import FlightAction


def _normal_telemetry(**overrides):
    base = {
        "lat": 28.35, "lon": 83.88, "altitude_m": 2500,
        "battery_pct": 70, "drain_rate": 1.5,
        "wind_speed": 8.0, "wind_dir": 270,
        "temp_c": 5, "payload_type": "oxytocin",
    }
    base.update(overrides)
    return base


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = KairosAnomalyDetector()

    def test_normal_telemetry_no_anomalies(self):
        t = _normal_telemetry()
        anomalies = self.detector.detect_anomalies(t)
        self.assertEqual(anomalies, [])

    def test_high_drain_rate_warning(self):
        t = _normal_telemetry(drain_rate=4.0)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("drain" in a.lower() for a in anomalies))

    def test_critical_drain_rate(self):
        t = _normal_telemetry(drain_rate=5.5)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("CRITICAL" in a for a in anomalies))

    def test_severe_wind_detected(self):
        t = _normal_telemetry(wind_speed=16.0)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("wind" in a.lower() for a in anomalies))

    def test_extreme_wind_critical(self):
        t = _normal_telemetry(wind_speed=22.0)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("CRITICAL" in a for a in anomalies))

    def test_battery_critical_emergency(self):
        t = _normal_telemetry(battery_pct=10.0)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("EMERGENCY" in a for a in anomalies))

    def test_combined_battery_wind_critical(self):
        t = _normal_telemetry(battery_pct=25.0, wind_speed=14.0)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("depletion" in a.lower() or "CRITICAL" in a for a in anomalies))

    def test_icing_risk_detected(self):
        t = _normal_telemetry(temp_c=-10.0, altitude_m=4000)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("icing" in a.lower() for a in anomalies))

    def test_altitude_ceiling_warning(self):
        t = _normal_telemetry(altitude_m=5300)
        anomalies = self.detector.detect_anomalies(t)
        self.assertTrue(any("ceiling" in a.lower() for a in anomalies))


class TestSeverityLevels(unittest.TestCase):

    def setUp(self):
        self.detector = KairosAnomalyDetector()

    def test_normal_severity_is_info(self):
        t = _normal_telemetry()
        severity = self.detector.get_severity(t)
        self.assertEqual(severity, AnomalySeverity.INFO.value)

    def test_low_battery_warning_severity(self):
        t = _normal_telemetry(battery_pct=25.0)
        severity = self.detector.get_severity(t)
        self.assertIn(severity, [AnomalySeverity.WARNING.value, AnomalySeverity.CRITICAL.value])

    def test_critical_battery_emergency(self):
        t = _normal_telemetry(battery_pct=10.0)
        severity = self.detector.get_severity(t)
        self.assertEqual(severity, AnomalySeverity.EMERGENCY.value)

    def test_extreme_wind_plus_low_battery_critical(self):
        t = _normal_telemetry(battery_pct=25.0, wind_speed=18.0)
        severity = self.detector.get_severity(t)
        self.assertIn(severity, [AnomalySeverity.CRITICAL.value, AnomalySeverity.EMERGENCY.value])


class TestActionRecommendation(unittest.TestCase):

    def setUp(self):
        self.detector = KairosAnomalyDetector()

    def test_normal_recommends_continue(self):
        t = _normal_telemetry()
        action = self.detector.recommend_action(t)
        self.assertEqual(action, FlightAction.CONTINUE.value)

    def test_emergency_battery_recommends_land(self):
        t = _normal_telemetry(battery_pct=8.0)
        action = self.detector.recommend_action(t)
        self.assertIn(action, [FlightAction.LAND.value, FlightAction.DIVERT.value])

    def test_extreme_wind_recommends_divert(self):
        t = _normal_telemetry(wind_speed=22.0, battery_pct=35.0)
        action = self.detector.recommend_action(t)
        self.assertIn(action, [FlightAction.DIVERT.value, FlightAction.RTL.value, FlightAction.LAND.value])

    def test_risk_score_normal_low(self):
        t = _normal_telemetry()
        score = self.detector.get_risk_score(t)
        self.assertLess(score, 0.3)

    def test_risk_score_critical_high(self):
        t = _normal_telemetry(battery_pct=10.0, wind_speed=22.0, altitude_m=4500, temp_c=-18.0)
        score = self.detector.get_risk_score(t)
        self.assertGreater(score, 0.5)

    def test_risk_score_bounded_0_to_1(self):
        t = _normal_telemetry(battery_pct=5.0, wind_speed=25.0, altitude_m=5500, temp_c=-20.0)
        score = self.detector.get_risk_score(t)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
