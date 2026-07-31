"""
The Kairos Engine - Automated Integration & Unit Test Suite
"""

import os
import unittest
from src.config import RISK_MODEL_PATH
from src.parser import GemmaParser
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor
from build_crash_predictor import assess_risk


class TestKairosEngine(unittest.TestCase):
    def test_gemma4_parser_dsl(self):
        sample_dsl = '<|tool_call|get_delivery_priority{payload_type:<|"|>oxytocin<|"|>}<tool_call|>'
        parsed = GemmaParser.extract_tool_calls(sample_dsl)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "get_delivery_priority")
        self.assertEqual(parsed[0]["arguments"]["payload_type"], "oxytocin")

    def test_tool_definitions_schema(self):
        tools = get_tool_definitions()
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("assess_risk", tool_names)
        self.assertIn("compute_energy_aware_route", tool_names)

    def test_xgb_risk_assessment_critical(self):
        res = assess_risk(battery_pct=0.20, wind_speed_ms=18.0, altitude_m=4000, proposed_action="RTL")
        self.assertIn("crash_probability", res)
        self.assertIn("risk_level", res)
        self.assertEqual(res["risk_level"], "Critical")

    def test_xgb_risk_assessment_acceptable(self):
        res = assess_risk(battery_pct=0.55, wind_speed_ms=5.0, altitude_m=1500, proposed_action="DIVERT")
        self.assertEqual(res["risk_level"], "Acceptable")

    def test_tool_executor_routing(self):
        executor = ToolExecutor()
        out = executor.execute("get_delivery_priority", {"payload_type": "oxytocin"})
        self.assertEqual(out, "critical")


if __name__ == "__main__":
    unittest.main()
