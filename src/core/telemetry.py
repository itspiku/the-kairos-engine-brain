"""
The Kairos Engine - Structured Dataclasses for Telemetry, Responses, and Events

TelemetryReport is the canonical in-flight state container passed through the
anomaly detector, LLM prompt formatter, and engine decision loop.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TelemetryReport:
    """
    Canonical in-flight telemetry container.

    All fields are named to match the telemetry prompt format used
    by format_telemetry_prompt() in src/utils/formatter.py.
    """
    lat: float
    lon: float
    altitude_m: float
    battery_pct: float
    drain_rate: float
    wind_speed: float
    wind_dir: float
    temp_c: float
    payload_type: str

    # Optional fields with defaults
    eta_min: Optional[int] = None
    nearest_lz: Optional[str] = None
    anomaly: Optional[str] = None
    priority: str = "standard"

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization and prompt formatting."""
        return asdict(self)
