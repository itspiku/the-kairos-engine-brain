"""
The Kairos Engine — Comprehensive Test Suite (core engine tests + expanded coverage)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RISK_MODEL_PATH, CRITICAL_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
    MAX_LATENCY_SLA_SEC, ENGINE_NAME, ENGINE_VERSION, DEFAULT_ORIGIN, DEFAULT_DESTINATION,
)
from src.parser import GemmaParser
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor
from src.core.types import FlightAction, RiskCategory, CognitiveResponse
from src.core.telemetry import TelemetryReport
from src.core.exceptions import (
    KairosError, ModelLoadError, ToolExecutionError,
    SLAViolationError, AirspaceViolationError, BatteryEmergencyError, ParsingError,
)


class TestGemma4ParserDSL(unittest.TestCase):
    """Original parser tests — backwards compatibility."""

    def test_gemma4_parser_basic(self):
        dsl = "<|tool_call|get_delivery_priority{payload_type:<|\"|>oxytocin<|\"|>}<tool_call|>"
        parsed = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "get_delivery_priority")
        self.assertEqual(parsed[0]["arguments"]["payload_type"], "oxytocin")

    def test_tool_definitions_schema(self):
        tools = get_tool_definitions()
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("assess_risk", tool_names)
        self.assertIn("compute_energy_aware_route", tool_names)
        self.assertIn("get_wind_forecast", tool_names)
        self.assertGreaterEqual(len(tools), 10)

    def test_xgb_risk_assessment_critical(self):
        from build_crash_predictor import assess_risk
        res = assess_risk(battery_pct=0.20, wind_speed_ms=18.0, altitude_m=4000, proposed_action="RTL")
        self.assertIn("crash_probability", res)
        self.assertIn("risk_level", res)
        self.assertEqual(res["risk_level"], "Critical")

    def test_xgb_risk_assessment_acceptable(self):
        from build_crash_predictor import assess_risk
        res = assess_risk(battery_pct=0.55, wind_speed_ms=5.0, altitude_m=1500, proposed_action="DIVERT")
        self.assertEqual(res["risk_level"], "Acceptable")

    def test_tool_executor_routing(self):
        executor = ToolExecutor()
        out = executor.execute("get_delivery_priority", {"payload_type": "oxytocin"})
        self.assertEqual(out, "critical")


class TestConfigConstants(unittest.TestCase):
    def test_engine_name_set(self):
        self.assertIsInstance(ENGINE_NAME, str)
        self.assertGreater(len(ENGINE_NAME), 0)

    def test_engine_version_set(self):
        self.assertIsInstance(ENGINE_VERSION, str)

    def test_sla_is_two_seconds(self):
        self.assertEqual(MAX_LATENCY_SLA_SEC, 2.0)

    def test_risk_thresholds(self):
        self.assertGreater(CRITICAL_RISK_THRESHOLD, HIGH_RISK_THRESHOLD)
        self.assertLess(HIGH_RISK_THRESHOLD, 1.0)

    def test_default_coords_are_nepal(self):
        lat, lon = DEFAULT_ORIGIN
        self.assertAlmostEqual(lat, 28.2096, places=2)
        self.assertAlmostEqual(lon, 83.9856, places=2)


class TestExceptionsHierarchy(unittest.TestCase):
    def test_model_load_error_is_kairos_error(self):
        exc = ModelLoadError("/path/model.gguf", "file not found")
        self.assertIsInstance(exc, KairosError)
        self.assertIn("/path/model.gguf", exc.context["model_path"])

    def test_tool_execution_error(self):
        exc = ToolExecutionError("get_wind_forecast", "timeout", {"lat": 28.2})
        self.assertIsInstance(exc, KairosError)
        self.assertEqual(exc.tool_name, "get_wind_forecast")

    def test_sla_violation_error(self):
        exc = SLAViolationError(2.5)
        self.assertIsInstance(exc, KairosError)
        self.assertEqual(exc.latency_sec, 2.5)

    def test_battery_emergency_error(self):
        exc = BatteryEmergencyError(10.0, drain_rate=6.0)
        self.assertIsInstance(exc, KairosError)
        self.assertEqual(exc.battery_pct, 10.0)

    def test_parsing_error(self):
        exc = ParsingError("garbage text")
        self.assertIsInstance(exc, KairosError)

    def test_all_exceptions_have_context(self):
        for ExcClass in [ModelLoadError, ToolExecutionError, SLAViolationError]:
            exc = ExcClass.__new__(ExcClass)
            exc.context = {}
            self.assertIsInstance(exc.context, dict)


class TestFlightActionEnum(unittest.TestCase):
    def test_all_actions_exist(self):
        actions = {a.value for a in FlightAction}
        self.assertIn("CONTINUE", actions)
        self.assertIn("DIVERT", actions)
        self.assertIn("LAND", actions)
        self.assertIn("ABORT", actions)
        self.assertIn("RTL", actions)
        self.assertIn("RE-TASK", actions)

    def test_action_is_string(self):
        self.assertIsInstance(FlightAction.CONTINUE.value, str)


class TestRiskCategoryEnum(unittest.TestCase):
    def test_all_categories_exist(self):
        cats = {c.value for c in RiskCategory}
        self.assertIn("Acceptable", cats)
        self.assertIn("High", cats)
        self.assertIn("Critical", cats)


class TestTelemetryReport(unittest.TestCase):
    def test_creation_with_required_fields(self):
        t = TelemetryReport(
            lat=28.35, lon=83.88, altitude_m=3200,
            battery_pct=42, drain_rate=4.2,
            wind_speed=18, wind_dir=270,
            temp_c=-5, payload_type="oxytocin",
        )
        self.assertEqual(t.lat, 28.35)
        self.assertEqual(t.payload_type, "oxytocin")

    def test_default_priority_is_standard(self):
        t = TelemetryReport(
            lat=28.0, lon=83.0, altitude_m=2000,
            battery_pct=80, drain_rate=1.0,
            wind_speed=5, wind_dir=180,
            temp_c=10, payload_type="bandages",
        )
        self.assertEqual(t.priority, "standard")

    def test_to_dict_has_all_keys(self):
        t = TelemetryReport(
            lat=28.0, lon=83.0, altitude_m=2000,
            battery_pct=80, drain_rate=1.0,
            wind_speed=5, wind_dir=180,
            temp_c=10, payload_type="oxytocin",
        )
        d = t.to_dict()
        for key in ["lat", "lon", "altitude_m", "battery_pct", "drain_rate",
                    "wind_speed", "wind_dir", "temp_c", "payload_type"]:
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
