"""
Core engine, telemetry models, types, and exception hierarchy.
"""

from src.core.engine import KairosEngine
from src.core.telemetry import TelemetryReport
from src.core.types import FlightAction, RiskCategory, CognitiveResponse
from src.core.exceptions import (
    KairosError, ModelLoadError, ToolExecutionError,
    SLAViolationError, AirspaceViolationError, BatteryEmergencyError,
)

__all__ = [
    "KairosEngine", "TelemetryReport",
    "FlightAction", "RiskCategory", "CognitiveResponse",
    "KairosError", "ModelLoadError", "ToolExecutionError",
    "SLAViolationError", "AirspaceViolationError", "BatteryEmergencyError",
]
