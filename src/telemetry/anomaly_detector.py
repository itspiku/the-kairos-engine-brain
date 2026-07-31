"""
The Kairos Engine - In-Flight Telemetry Anomaly Detector & Contingency Evaluator
"""

from typing import Dict, Any, List


class KairosAnomalyDetector:
    """Monitors telemetry stream for battery spikes, extreme headwinds, and altitude drops."""

    @staticmethod
    def detect_anomalies(telemetry: Dict[str, Any]) -> List[str]:
        anomalies = []
        battery = telemetry.get("battery_pct", 100)
        drain_rate = telemetry.get("drain_rate", 0)
        wind_speed = telemetry.get("wind_speed", 0)

        if drain_rate > 3.5:
            anomalies.append(f"Abnormal battery drain rate detected: {drain_rate}%/min (Threshold: 3.5%/min)")

        if wind_speed > 15.0:
            anomalies.append(f"Severe headwind speed detected: {wind_speed} m/s (Threshold: 15.0 m/s)")

        if battery < 25.0 and wind_speed > 12.0:
            anomalies.append("CRITICAL: Battery depletion risk under high headwind condition")

        return anomalies


class ContingencyEvaluator:
    @staticmethod
    def evaluate_landing_zones(current_pos: List[float], radius_m: float = 2000) -> List[Dict[str, Any]]:
        return [
            {"id": "LZ_01", "name": "High-Altitude Base LZ", "pos": [28.3500, 83.8800], "flatness_score": 0.94, "dist_m": 850},
            {"id": "LZ_02", "name": "Jomsom Emergency Airstrip", "pos": [28.7800, 83.7200], "flatness_score": 0.89, "dist_m": 1400},
            {"id": "LZ_03", "name": "Thorong Phedi Shelter Field", "pos": [28.7900, 83.9400], "flatness_score": 0.96, "dist_m": 1650}
        ]
