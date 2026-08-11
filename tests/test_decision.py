"""
The Kairos Engine — Decision Extraction & Safe-Fallback Test Suite

Covers two defects in the in-flight decision path:

1. Actions were matched by substring, so "continue to the landing zone" resolved
   to LAND because LAND appears inside "landing".
2. When nothing matched, the decision defaulted to CONTINUE — flying on into the
   condition that triggered the emergency. An unparseable response is the worst
   case in which to assume it is safe to continue.

Also the first coverage of KairosEngine.in_flight_decision, which had none.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import GemmaParser
from src.core.types import FlightAction
from src.core.telemetry import TelemetryReport


class TestDecisionExtraction(unittest.TestCase):

    def test_landing_zone_does_not_resolve_to_land(self):
        """The substring bug: LAND is inside 'landing'."""
        self.assertEqual(
            GemmaParser.extract_decision("Continue to the landing zone LZ_03 and hold."),
            "CONTINUE",
        )

    def test_plain_action_mentions(self):
        for text, expected in [
            ("I recommend we DIVERT to LZ_02 immediately.", "DIVERT"),
            ("Battery critical, LAND now.", "LAND"),
            ("Abort the mission.", "ABORT"),
            ("Return to launch: RTL.", "RTL"),
        ]:
            self.assertEqual(GemmaParser.extract_decision(text), expected, text)

    def test_explicit_declaration_wins_over_prose(self):
        text = "We could DIVERT or LAND.\nDECISION: CONTINUE\nThe route remains viable."
        self.assertEqual(GemmaParser.extract_decision(text), "CONTINUE")

    def test_markdown_bold_declaration(self):
        self.assertEqual(GemmaParser.extract_decision("**Decision:** DIVERT"), "DIVERT")

    def test_final_decision_with_dash(self):
        self.assertEqual(GemmaParser.extract_decision("Final Decision - RTL"), "RTL")

    def test_ambiguous_prose_prefers_the_more_cautious_action(self):
        text = "Considered CONTINUE, but conditions demand we LAND now."
        self.assertEqual(GemmaParser.extract_decision(text), "LAND")

    def test_hyphenated_action(self):
        self.assertEqual(
            GemmaParser.extract_decision("Hand off the payload. RE-TASK confirmed."),
            "RE-TASK",
        )

    def test_no_action_returns_none(self):
        self.assertIsNone(GemmaParser.extract_decision("The weather is fine, terrain clear."))

    def test_empty_returns_default(self):
        self.assertIsNone(GemmaParser.extract_decision(""))
        self.assertEqual(GemmaParser.extract_decision("", default="DIVERT"), "DIVERT")

    def test_every_extracted_value_is_a_valid_flight_action(self):
        valid = {a.value for a in FlightAction}
        for text in ["DIVERT now", "LAND", "ABORT", "RTL", "RE-TASK", "CONTINUE"]:
            self.assertIn(GemmaParser.extract_decision(text), valid)


class TestEngineSafeFallback(unittest.TestCase):
    """KairosEngine.in_flight_decision must never fail open to CONTINUE."""

    EMERGENCY = TelemetryReport(
        lat=28.35, lon=83.88, altitude_m=4200,
        battery_pct=8.0, drain_rate=6.5,
        wind_speed=22.0, wind_dir=270,
        temp_c=-16.0, payload_type="oxytocin",
        anomaly="Battery critical, severe headwind",
    )

    def _engine_with_llm_saying(self, content):
        """Build a KairosEngine whose LLM returns `content`, without loading a model."""
        with patch("src.core.engine.GemmaLLMEngine") as MockLLM:
            instance = MockLLM.return_value
            instance.agentic_generate.return_value = {
                "content": content, "raw": content, "thinking": "",
                "tool_calls": [], "all_tool_calls": [], "all_tool_results": [],
                "latency_sec": 0.5, "iterations": 1,
            }
            from src.core.engine import KairosEngine
            return KairosEngine(model_path="test-model-not-loaded")

    def test_unparseable_output_does_not_continue(self):
        engine = self._engine_with_llm_saying("The mountains are beautiful today.")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertNotEqual(
            result["decision"], FlightAction.CONTINUE.value,
            "an unreadable LLM response must not resolve to CONTINUE in an emergency",
        )

    def test_unparseable_output_uses_deterministic_recommendation(self):
        engine = self._engine_with_llm_saying("The mountains are beautiful today.")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertEqual(result["decision"], result["recommended_action"])

    def test_emergency_fallback_is_land_or_divert(self):
        engine = self._engine_with_llm_saying("no actionable content here")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertIn(result["decision"],
                      [FlightAction.LAND.value, FlightAction.DIVERT.value])

    def test_parsed_decision_is_honoured(self):
        engine = self._engine_with_llm_saying("DECISION: DIVERT\nHeading to LZ_02.")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertEqual(result["decision"], FlightAction.DIVERT.value)

    def test_landing_zone_phrasing_survives_the_engine_path(self):
        engine = self._engine_with_llm_saying("Continue past the landing zone; margins hold.")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertEqual(result["decision"], FlightAction.CONTINUE.value)

    def test_anomalies_are_reported(self):
        engine = self._engine_with_llm_saying("DECISION: LAND")
        result = engine.in_flight_decision(self.EMERGENCY)
        self.assertGreater(len(result["anomalies"]), 0)
        self.assertEqual(result["severity"], "EMERGENCY")


if __name__ == "__main__":
    unittest.main()
