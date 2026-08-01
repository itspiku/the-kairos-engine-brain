"""
The Kairos Engine — Telemetry Dataclass Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.telemetry import TelemetryReport


SAMPLE = dict(
    lat=28.35, lon=83.88, altitude_m=3200,
    battery_pct=42.0, drain_rate=4.2,
    wind_speed=18.0, wind_dir=270.0,
    temp_c=-5.0, payload_type="oxytocin",
)


class TestTelemetryReport(unittest.TestCase):

    def test_creation_minimal(self):
        t = TelemetryReport(**SAMPLE)
        self.assertEqual(t.lat, 28.35)
        self.assertEqual(t.battery_pct, 42.0)

    def test_optional_fields_default_none(self):
        t = TelemetryReport(**SAMPLE)
        self.assertIsNone(t.eta_min)
        self.assertIsNone(t.nearest_lz)
        self.assertIsNone(t.anomaly)

    def test_default_priority_standard(self):
        t = TelemetryReport(**SAMPLE)
        self.assertEqual(t.priority, "standard")

    def test_custom_priority(self):
        t = TelemetryReport(**SAMPLE, priority="critical")
        self.assertEqual(t.priority, "critical")

    def test_optional_fields_set(self):
        t = TelemetryReport(**SAMPLE, eta_min=25, nearest_lz="LZ_03",
                            anomaly="High drain rate")
        self.assertEqual(t.eta_min, 25)
        self.assertEqual(t.nearest_lz, "LZ_03")
        self.assertEqual(t.anomaly, "High drain rate")

    def test_to_dict_returns_dict(self):
        t = TelemetryReport(**SAMPLE)
        d = t.to_dict()
        self.assertIsInstance(d, dict)

    def test_to_dict_has_all_required_keys(self):
        t = TelemetryReport(**SAMPLE)
        d = t.to_dict()
        required_keys = [
            "lat", "lon", "altitude_m", "battery_pct", "drain_rate",
            "wind_speed", "wind_dir", "temp_c", "payload_type",
            "eta_min", "nearest_lz", "anomaly", "priority",
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_to_dict_values_match(self):
        t = TelemetryReport(**SAMPLE, eta_min=30)
        d = t.to_dict()
        self.assertEqual(d["lat"], 28.35)
        self.assertEqual(d["battery_pct"], 42.0)
        self.assertEqual(d["eta_min"], 30)

    def test_roundtrip_dict(self):
        t = TelemetryReport(**SAMPLE)
        d = t.to_dict()
        # Should be able to reconstruct from dict
        t2 = TelemetryReport(
            lat=d["lat"], lon=d["lon"], altitude_m=d["altitude_m"],
            battery_pct=d["battery_pct"], drain_rate=d["drain_rate"],
            wind_speed=d["wind_speed"], wind_dir=d["wind_dir"],
            temp_c=d["temp_c"], payload_type=d["payload_type"],
        )
        self.assertEqual(t2.lat, t.lat)
        self.assertEqual(t2.battery_pct, t.battery_pct)


if __name__ == "__main__":
    unittest.main()
