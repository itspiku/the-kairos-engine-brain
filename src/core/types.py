"""
The Kairos Engine - Domain Types: Enums and Dataclasses

Defines the canonical action/category types used across engine,
telemetry, anomaly detection, and routing modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class FlightAction(str, Enum):
    """
    Available cognitive autopilot flight actions.
    Used by anomaly detector, engine, and LLM response parsing.
    """
    CONTINUE = "CONTINUE"   # Maintain current route and parameters
    RTL      = "RTL"        # Return-to-launch (home base)
    DIVERT   = "DIVERT"     # Divert to nearest contingency LZ
    LAND     = "LAND"       # Immediate emergency landing
    ABORT    = "ABORT"      # Abort mission, controlled descent
    RE_TASK  = "RE-TASK"    # Hand off mission to secondary drone


class RiskCategory(str, Enum):
    """XGBoost risk model output categories."""
    ACCEPTABLE = "Acceptable"
    HIGH       = "High"
    CRITICAL   = "Critical"


@dataclass
class CognitiveResponse:
    """Structured output from a Gemma 4 LLM inference call."""
    content: str
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_sec: float = 0.0
    meets_sla: bool = True
    raw: str = ""
    iterations: int = 1
    all_tool_calls: List[Dict] = field(default_factory=list)
    all_tool_results: List[Dict] = field(default_factory=list)
