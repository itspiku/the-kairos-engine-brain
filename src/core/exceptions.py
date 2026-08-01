"""
The Kairos Engine - Structured Exception Hierarchy

All custom exceptions inherit from KairosError, providing structured
context for logging, debugging, and audit trail entries.
"""

from typing import Any, Dict, Optional


class KairosError(Exception):
    """Base exception for all Kairos Engine errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class ModelLoadError(KairosError):
    """Raised when the Gemma 4 GGUF or XGBoost model fails to load."""

    def __init__(self, model_path: str, reason: str = ""):
        super().__init__(
            f"Failed to load model '{model_path}': {reason}",
            context={"model_path": model_path, "reason": reason},
        )
        self.model_path = model_path
        self.reason = reason


class ToolExecutionError(KairosError):
    """Raised when a cognitive tool call fails during execution."""

    def __init__(self, tool_name: str, reason: str = "", arguments: Optional[Dict] = None):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {reason}",
            context={"tool_name": tool_name, "reason": reason, "arguments": arguments or {}},
        )
        self.tool_name = tool_name
        self.reason = reason


class SLAViolationError(KairosError):
    """Raised when a decision exceeds the 2-second latency SLA."""

    def __init__(self, latency_sec: float, sla_sec: float = 2.0):
        super().__init__(
            f"SLA violation: decision took {latency_sec:.2f}s (limit: {sla_sec}s)",
            context={"latency_sec": latency_sec, "sla_sec": sla_sec},
        )
        self.latency_sec = latency_sec
        self.sla_sec = sla_sec


class AirspaceViolationError(KairosError):
    """Raised when a proposed route intersects a restricted airspace zone."""

    def __init__(self, zone_id: str, zone_name: str = ""):
        super().__init__(
            f"Airspace violation: zone '{zone_id}' ({zone_name}) intersects route",
            context={"zone_id": zone_id, "zone_name": zone_name},
        )
        self.zone_id = zone_id


class BatteryEmergencyError(KairosError):
    """Raised when battery reaches emergency threshold requiring immediate action."""

    def __init__(self, battery_pct: float, drain_rate: float = 0.0):
        super().__init__(
            f"Battery emergency: {battery_pct:.1f}% remaining (drain: {drain_rate:.1f}%/min)",
            context={"battery_pct": battery_pct, "drain_rate": drain_rate},
        )
        self.battery_pct = battery_pct
        self.drain_rate = drain_rate


class ParsingError(KairosError):
    """Raised when the Gemma 4 DSL parser cannot parse LLM output."""

    def __init__(self, raw_text: str, reason: str = ""):
        preview = raw_text[:100] if raw_text else ""
        super().__init__(
            f"Failed to parse LLM output: {reason}. Preview: {preview!r}",
            context={"raw_preview": preview, "reason": reason},
        )


class RouteInfeasibleError(KairosError):
    """Raised when no feasible route exists given current energy constraints."""

    def __init__(self, reason: str = "", required_wh: float = 0, available_wh: float = 0):
        super().__init__(
            f"Route infeasible: {reason} (required: {required_wh:.0f}Wh, available: {available_wh:.0f}Wh)",
            context={"reason": reason, "required_wh": required_wh, "available_wh": available_wh},
        )
        self.required_wh = required_wh
        self.available_wh = available_wh
