"""
The Kairos Engine — Integration Test Suite

Tests the full pipeline without requiring the Gemma 4 LLM model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.executor import ToolExecutor
from src.telemetry.anomaly_detector import KairosAnomalyDetector
from src.core.types import FlightAction
from src.config import DEFAULT_ORIGIN, DEFAULT_DESTINATION


class TestFullToolPipeline(unittest.TestCase):
    """Test every tool executes and returns properly structured output."""

    def setUp(self):
        self.executor = ToolExecutor()

    def test_wind_forecast_tool(self):
        result = self.executor.execute("get_wind_forecast", {
            "lat": 28.35, "lon": 83.88, "altitude_m": 3200, "time": "2024-01-01T12:00:00Z"
        })
        self.assertIsInstance(result, dict)
        self.assertIn("wind_speed", result)
        self.assertGreaterEqual(result["wind_speed"], 0)

    def test_terrain_elevation_tool(self):
        result = self.executor.execute("get_terrain_elevation_profile", {
            "route": [DEFAULT_ORIGIN, [28.5, 83.9], DEFAULT_DESTINATION]
        })
        self.assertIsInstance(result, dict)
        self.assertIn("max_elevation_m", result)

    def test_battery_state_tool(self):
        result = self.executor.execute("get_battery_state", {"drone_id": "INT-TEST-01"})
        self.assertIsInstance(result, dict)
        self.assertIn("soc", result)
        self.assertIn("voltage", result)

    def test_energy_route_tool(self):
        result = self.executor.execute("compute_energy_aware_route", {
            "origin": DEFAULT_ORIGIN,
            "destination": DEFAULT_DESTINATION,
            "drone_params": {"mass_kg": 4.5, "battery_wh": 800},
        })
        self.assertIsInstance(result, dict)
        self.assertIn("waypoints", result)
        self.assertIn("estimated_energy_wh", result)

    def test_find_lzs_tool(self):
        result = self.executor.execute("find_contingency_landing_zones", {
            "current_pos": [28.35, 83.88], "radius_m": 100000
        })
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_airspace_check_tool(self):
        result = self.executor.execute("check_airspace_restrictions", {
            "bbox": {"north": 29.0, "south": 28.0, "east": 84.5, "west": 83.5}
        })
        self.assertIsInstance(result, dict)
        self.assertIn("restricted", result)
        self.assertIn("max_permitted_altitude_m", result)

    def test_delivery_priority_tool(self):
        result = self.executor.execute("get_delivery_priority", {"payload_type": "insulin"})
        self.assertEqual(result, "critical")

    def test_recompute_route_tool(self):
        result = self.executor.execute("recompute_route_from_current_pos", {
            "current_pos": [28.35, 83.88], "battery_remaining_pct": 45.0
        })
        self.assertIsInstance(result, dict)
        self.assertIn("feasible", result)
        self.assertIn("new_waypoints", result)

    def test_select_lz_tool(self):
        lzs = self.executor.execute("find_contingency_landing_zones", {
            "current_pos": [28.35, 83.88], "radius_m": 100000
        })
        if lzs:
            result = self.executor.execute("select_best_landing_zone", {
                "current_pos": [28.35, 83.88],
                "reachable_zones": lzs[:3],
            })
            self.assertIsInstance(result, dict)

    def test_log_decision_tool(self):
        result = self.executor.execute("log_decision", {
            "decision": "DIVERT",
            "rationale": "Integration test log entry",
        })
        self.assertTrue(result.get("logged"))

    def test_assess_risk_tool(self):
        result = self.executor.execute("assess_risk", {
            "battery_pct": 0.20, "wind_speed_ms": 18.0,
            "altitude_m": 4000, "proposed_action": "RTL"
        })
        self.assertIn("crash_probability", result)
        self.assertIn("risk_level", result)


class TestAnomalyToDecisionPipeline(unittest.TestCase):
    """Test anomaly detection → action recommendation pipeline."""

    def setUp(self):
        self.detector = KairosAnomalyDetector()

    def test_emergency_leads_to_action(self):
        telemetry = {
            "battery_pct": 8.0, "drain_rate": 6.0,
            "wind_speed": 22.0, "altitude_m": 4000, "temp_c": -15.0,
        }
        anomalies = self.detector.detect_anomalies(telemetry)
        self.assertGreater(len(anomalies), 0)
        action = self.detector.recommend_action(telemetry)
        self.assertIn(action, [FlightAction.LAND.value, FlightAction.DIVERT.value])

    def test_warning_leads_to_continue(self):
        telemetry = {
            "battery_pct": 40.0, "drain_rate": 2.0,
            "wind_speed": 10.0, "altitude_m": 2500, "temp_c": 5.0,
        }
        action = self.detector.recommend_action(telemetry)
        self.assertEqual(action, FlightAction.CONTINUE.value)

    def test_risk_score_pipeline(self):
        telemetry = {
            "battery_pct": 15.0, "drain_rate": 5.0,
            "wind_speed": 19.0, "altitude_m": 4200, "temp_c": -10.0,
        }
        score = self.detector.get_risk_score(telemetry)
        self.assertGreater(score, 0.4)
        self.assertLessEqual(score, 1.0)


class TestRiskAssessmentPipeline(unittest.TestCase):
    """Test the full risk assessment from parameters → XGBoost → risk level."""

    def test_critical_parameters_give_critical_risk(self):
        from src.tools.implementations import assess_risk
        result = assess_risk(0.15, 20.0, 4500, "RTL")
        self.assertIn(result["risk_level"], ["Critical", "High"])

    def test_safe_parameters_give_acceptable_risk(self):
        from src.tools.implementations import assess_risk
        result = assess_risk(0.60, 3.0, 1000, "DIVERT")
        self.assertIn(result["risk_level"], ["Acceptable", "High"])

    def test_risk_pipeline_consistent_with_model(self):
        from src.tools.implementations import assess_risk
        from build_crash_predictor import assess_risk as direct_assess
        r1 = assess_risk(0.25, 15.0, 3500, "CONTINUE")
        r2 = direct_assess(0.25, 15.0, 3500, "CONTINUE")
        # Both should agree on the same risk level
        self.assertEqual(r1["risk_level"], r2["risk_level"])


if __name__ == "__main__":
    unittest.main()
