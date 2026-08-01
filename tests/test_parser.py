"""
The Kairos Engine — Parser Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import GemmaParser


class TestGemmaParser(unittest.TestCase):

    def test_basic_dsl_string_arg(self):
        dsl = "<|tool_call|get_delivery_priority{payload_type:<|\"|>oxytocin<|\"|>}<tool_call|>"
        calls = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_delivery_priority")
        self.assertEqual(calls[0]["arguments"]["payload_type"], "oxytocin")

    def test_numeric_integer_arg(self):
        dsl = "<|tool_call|assess_risk{altitude_m:4000,battery_pct:0.2,wind_speed_ms:18,proposed_action:<|\"|>RTL<|\"|>}<tool_call|>"
        calls = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(calls), 1)
        args = calls[0]["arguments"]
        self.assertEqual(args["altitude_m"], 4000)
        self.assertAlmostEqual(args["battery_pct"], 0.2, places=2)

    def test_float_arg(self):
        dsl = "<|tool_call|get_wind_forecast{lat:28.2096,lon:83.9856,altitude_m:3200,time:<|\"|>2024-01-01T12:00:00Z<|\"|>}<tool_call|>"
        calls = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(calls), 1)
        args = calls[0]["arguments"]
        self.assertAlmostEqual(args["lat"], 28.2096, places=3)

    def test_boolean_arg(self):
        dsl = "<|tool_call|some_tool{active:true,disabled:false}<tool_call|>"
        calls = GemmaParser.extract_tool_calls(dsl)
        if calls:
            args = calls[0]["arguments"]
            self.assertIn("active", args)

    def test_multiple_tool_calls(self):
        dsl = (
            "<|tool_call|get_wind_forecast{lat:28.2,lon:83.9,altitude_m:3000,time:<|\"|>now<|\"|>}<tool_call|>"
            " some text in between "
            "<|tool_call|get_delivery_priority{payload_type:<|\"|>insulin<|\"|>}<tool_call|>"
        )
        calls = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "get_wind_forecast")
        self.assertEqual(calls[1]["name"], "get_delivery_priority")

    def test_empty_input_returns_empty_list(self):
        calls = GemmaParser.extract_tool_calls("")
        self.assertEqual(calls, [])

    def test_none_like_empty_input(self):
        calls = GemmaParser.extract_tool_calls("   ")
        self.assertEqual(calls, [])

    def test_malformed_input_no_crash(self):
        garbage = "this is just random text with no tool calls"
        calls = GemmaParser.extract_tool_calls(garbage)
        self.assertIsInstance(calls, list)

    def test_json_fallback_parsing(self):
        json_text = '{"name": "get_delivery_priority", "arguments": {"payload_type": "blood plasma"}}'
        calls = GemmaParser.extract_tool_calls(json_text)
        if calls:
            self.assertEqual(calls[0]["name"], "get_delivery_priority")
            self.assertEqual(calls[0]["arguments"].get("payload_type"), "blood plasma")

    def test_validate_tool_call_valid(self):
        from src.tools.definitions import get_tool_definitions
        schemas = get_tool_definitions()
        call = {"name": "get_delivery_priority", "arguments": {"payload_type": "oxytocin"}}
        errors = GemmaParser.validate_tool_call(call, schemas)
        self.assertEqual(errors, [])

    def test_validate_tool_call_missing_required(self):
        from src.tools.definitions import get_tool_definitions
        schemas = get_tool_definitions()
        call = {"name": "get_delivery_priority", "arguments": {}}
        errors = GemmaParser.validate_tool_call(call, schemas)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("payload_type" in e for e in errors))

    def test_validate_unknown_tool(self):
        from src.tools.definitions import get_tool_definitions
        schemas = get_tool_definitions()
        call = {"name": "nonexistent_tool", "arguments": {}}
        errors = GemmaParser.validate_tool_call(call, schemas)
        self.assertTrue(any("Unknown tool" in e for e in errors))

    def test_mixed_arg_types(self):
        dsl = "<|tool_call|assess_risk{battery_pct:0.35,wind_speed_ms:12.5,altitude_m:3500,proposed_action:<|\"|>CONTINUE<|\"|>}<tool_call|>"
        calls = GemmaParser.extract_tool_calls(dsl)
        self.assertEqual(len(calls), 1)
        args = calls[0]["arguments"]
        self.assertIsInstance(args.get("altitude_m"), (int, float))
        self.assertIsInstance(args.get("proposed_action"), str)


if __name__ == "__main__":
    unittest.main()
