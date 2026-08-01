"""
The Kairos Engine - In-Flight Telemetry Anomaly Detector & Contingency Evaluator

Production anomaly detection with severity levels, rate-of-change analysis,
combined risk scoring, icing detection, and action recommendation.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from src.core.types import FlightAction

logger = logging.getLogger("kairos.anomaly")


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class KairosAnomalyDetector:
    """Monitors telemetry stream for dangerous conditions with severity-graded alerts."""

    # Thresholds
    DRAIN_RATE_WARN = 3.5       # %/min
    DRAIN_RATE_CRIT = 5.0       # %/min
    WIND_WARN = 15.0            # m/s
    WIND_CRIT = 20.0            # m/s
    BATTERY_LOW = 30.0          # %
    BATTERY_CRIT = 15.0         # %
    TEMP_ICING = -8.0           # °C
    ALTITUDE_CEILING = 5200.0   # m

    def __init__(self):
        self._history: List[Dict] = []

    def detect_anomalies(self, telemetry: Dict[str, Any]) -> List[str]:
        """
        Detect all anomalies from a telemetry snapshot.
        Returns list of human-readable anomaly descriptions.
        """
        self._history.append(telemetry)
        anomalies = []

        battery = telemetry.get("battery_pct", 100)
        drain_rate = telemetry.get("drain_rate", 0)
        wind_speed = telemetry.get("wind_speed", 0)
        altitude = telemetry.get("altitude_m", 0)
        temp = telemetry.get("temp_c", 15)

        # Battery anomalies
        if drain_rate > self.DRAIN_RATE_CRIT:
            anomalies.append(
                f"CRITICAL: Extreme battery drain rate {drain_rate:.1f}%/min "
                f"(threshold: {self.DRAIN_RATE_CRIT}%/min)"
            )
        elif drain_rate > self.DRAIN_RATE_WARN:
            anomalies.append(
                f"WARNING: Abnormal battery drain rate {drain_rate:.1f}%/min "
                f"(threshold: {self.DRAIN_RATE_WARN}%/min)"
            )

        if battery < self.BATTERY_CRIT:
            anomalies.append(f"EMERGENCY: Battery critically low at {battery:.1f}%")
        elif battery < self.BATTERY_LOW:
            anomalies.append(f"WARNING: Battery low at {battery:.1f}%")

        # Wind anomalies
        if wind_speed > self.WIND_CRIT:
            anomalies.append(
                f"CRITICAL: Extreme headwind {wind_speed:.1f} m/s "
                f"(threshold: {self.WIND_CRIT} m/s)"
            )
        elif wind_speed > self.WIND_WARN:
            anomalies.append(
                f"WARNING: Severe headwind {wind_speed:.1f} m/s "
                f"(threshold: {self.WIND_WARN} m/s)"
            )

        # Combined battery + wind
        if battery < self.BATTERY_LOW and wind_speed > 12.0:
            anomalies.append(
                "CRITICAL: Battery depletion risk under high headwind condition"
            )

        # Icing risk
        if temp < self.TEMP_ICING and altitude > 3500:
            anomalies.append(
                f"WARNING: Icing risk — temperature {temp:.0f}°C at {altitude:.0f}m"
            )
        if temp < -15 and altitude > 4000:
            anomalies.append(
                f"CRITICAL: Severe icing conditions — {temp:.0f}°C at {altitude:.0f}m"
            )

        # Altitude ceiling
        if altitude > self.ALTITUDE_CEILING:
            anomalies.append(
                f"WARNING: Exceeding operational ceiling: {altitude:.0f}m "
                f"(limit: {self.ALTITUDE_CEILING:.0f}m)"
            )

        # Rate-of-change detection (need at least 2 data points)
        if len(self._history) >= 2:
            prev = self._history[-2]
            wind_delta = wind_speed - prev.get("wind_speed", wind_speed)
            if wind_delta > 5:
                anomalies.append(
                    f"INFO: Rapid wind speed increase: +{wind_delta:.1f} m/s between readings"
                )

            drain_delta = drain_rate - prev.get("drain_rate", drain_rate)
            if drain_delta > 1.5:
                anomalies.append(
                    f"WARNING: Battery drain accelerating: +{drain_delta:.1f}%/min increase"
                )

        return anomalies

    def get_severity(self, telemetry: Dict[str, Any]) -> str:
        """
        Calculate overall anomaly severity from telemetry.
        Returns highest severity level found.
        """
        anomalies = self.detect_anomalies(telemetry) if not self._history else []
        # Re-evaluate from current state
        battery = telemetry.get("battery_pct", 100)
        drain_rate = telemetry.get("drain_rate", 0)
        wind_speed = telemetry.get("wind_speed", 0)
        temp = telemetry.get("temp_c", 15)
        altitude = telemetry.get("altitude_m", 0)

        if battery < self.BATTERY_CRIT:
            return AnomalySeverity.EMERGENCY.value
        if (drain_rate > self.DRAIN_RATE_CRIT and wind_speed > self.WIND_WARN):
            return AnomalySeverity.EMERGENCY.value
        if (battery < self.BATTERY_LOW and wind_speed > self.WIND_WARN):
            return AnomalySeverity.CRITICAL.value
        if drain_rate > self.DRAIN_RATE_CRIT or wind_speed > self.WIND_CRIT:
            return AnomalySeverity.CRITICAL.value
        if temp < -15 and altitude > 4000:
            return AnomalySeverity.CRITICAL.value
        if (drain_rate > self.DRAIN_RATE_WARN or wind_speed > self.WIND_WARN or
                battery < self.BATTERY_LOW or temp < self.TEMP_ICING):
            return AnomalySeverity.WARNING.value

        return AnomalySeverity.INFO.value

    def recommend_action(self, telemetry: Dict[str, Any]) -> str:
        """
        Recommend a FlightAction based on anomaly severity and conditions.
        """
        severity = self.get_severity(telemetry)
        battery = telemetry.get("battery_pct", 100)
        wind_speed = telemetry.get("wind_speed", 0)

        if severity == AnomalySeverity.EMERGENCY.value:
            if battery < self.BATTERY_CRIT:
                return FlightAction.LAND.value
            return FlightAction.DIVERT.value

        if severity == AnomalySeverity.CRITICAL.value:
            if wind_speed > self.WIND_CRIT:
                return FlightAction.DIVERT.value
            return FlightAction.RTL.value

        if severity == AnomalySeverity.WARNING.value:
            return FlightAction.CONTINUE.value

        return FlightAction.CONTINUE.value

    def get_risk_score(self, telemetry: Dict[str, Any]) -> float:
        """
        Calculate a combined risk score from 0.0 (safe) to 1.0 (extreme danger).
        """
        score = 0.0
        battery = telemetry.get("battery_pct", 100)
        drain_rate = telemetry.get("drain_rate", 0)
        wind_speed = telemetry.get("wind_speed", 0)
        temp = telemetry.get("temp_c", 15)
        altitude = telemetry.get("altitude_m", 0)

        # Battery risk (0-0.35)
        if battery < 15:
            score += 0.35
        elif battery < 30:
            score += 0.20
        elif battery < 50:
            score += 0.08

        # Drain rate risk (0-0.20)
        if drain_rate > 5.0:
            score += 0.20
        elif drain_rate > 3.5:
            score += 0.10

        # Wind risk (0-0.25)
        if wind_speed > 20:
            score += 0.25
        elif wind_speed > 15:
            score += 0.15
        elif wind_speed > 10:
            score += 0.05

        # Icing risk (0-0.10)
        if temp < -15 and altitude > 4000:
            score += 0.10
        elif temp < -8 and altitude > 3500:
            score += 0.05

        # Altitude risk (0-0.10)
        if altitude > 5200:
            score += 0.10
        elif altitude > 4500:
            score += 0.05

        return min(score, 1.0)


class ContingencyEvaluator:
    """Evaluates and ranks contingency landing zones for emergency scenarios."""

    @staticmethod
    def evaluate_landing_zones(current_pos: List[float],
                               radius_m: float = 2000) -> List[Dict[str, Any]]:
        """
        Find and rank landing zones near the current position.
        Uses the full LZ database from implementations.
        """
        try:
            from src.tools.implementations import find_contingency_landing_zones
            return find_contingency_landing_zones(current_pos, radius_m)
        except ImportError:
            # Fallback to hardcoded LZs
            return [
                {"id": "LZ_01", "name": "Lete River Terrace",
                 "pos": [28.3500, 83.8800], "flatness": 0.94, "distance_m": 850},
                {"id": "LZ_02", "name": "Jomsom Emergency Airstrip",
                 "pos": [28.7800, 83.7200], "flatness": 0.97, "distance_m": 1400},
                {"id": "LZ_03", "name": "Thorong Phedi Shelter Field",
                 "pos": [28.7900, 83.9400], "flatness": 0.91, "distance_m": 1650},
            ]
