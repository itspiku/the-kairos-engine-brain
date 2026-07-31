"""
The Kairos Engine - Core Types, Enums & Action Constants
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class FlightAction(str, Enum):
    CONTINUE = "CONTINUE"
    RTL = "RTL"
    DIVERT = "DIVERT"
    LAND = "LAND"
    ABORT = "ABORT"
    RE_TASK = "RE-TASK"


class RiskCategory(str, Enum):
    ACCEPTABLE = "Acceptable"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class CognitiveResponse:
    raw: str
    content: str
    thinking: str
    tool_calls: List[Dict[str, Any]]
    latency_sec: float
    meets_sla: bool = True
