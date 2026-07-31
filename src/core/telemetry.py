"""
The Kairos Engine - Telemetry Data Models & In-Flight State Representation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class TelemetryReport:
    lat: float
    lon: float
    altitude_m: float
    battery_pct: float
    drain_rate: float
    wind_speed: float
    wind_dir: float
    temp_c: float
    payload_type: str
    eta_min: Optional[float] = None
    nearest_lz: Optional[str] = None
    anomaly: Optional[str] = None
    priority: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "altitude_m": self.altitude_m,
            "battery_pct": self.battery_pct,
            "drain_rate": self.drain_rate,
            "wind_speed": self.wind_speed,
            "wind_dir": self.wind_dir,
            "temp_c": self.temp_c,
            "payload_type": self.payload_type,
            "eta_min": self.eta_min,
            "nearest_lz": self.nearest_lz,
            "anomaly": self.anomaly,
            "priority": self.priority
        }
