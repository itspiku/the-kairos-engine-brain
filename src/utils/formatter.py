"""
The Kairos Engine - Output Formatters & Payload Serialization Utilities
"""

import json
from typing import Dict, Any


def format_json_pretty(data: Dict[str, Any]) -> str:
    """Format dictionary into pretty JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_telemetry_prompt(telemetry: Dict[str, Any]) -> str:
    """Formats telemetry dictionary into clean prompt text for Gemma 4."""
    return (
        f"TELEMETRY REPORT:\n"
        f"- Position: {telemetry.get('lat', 0):.4f}, {telemetry.get('lon', 0):.4f} @ {telemetry.get('altitude_m', 0)}m\n"
        f"- Battery: {telemetry.get('battery_pct', 0)}% (draining at {telemetry.get('drain_rate', 0)}%/min)\n"
        f"- Wind: {telemetry.get('wind_speed', 0)} m/s from {telemetry.get('wind_dir', 0)}°\n"
        f"- Temperature: {telemetry.get('temp_c', 0)}°C\n"
        f"- Payload: {telemetry.get('payload_type', 'medical')} (Priority: {telemetry.get('priority', 'standard')})\n"
        f"- Time to destination: {telemetry.get('eta_min', '?')} min\n"
        f"- Nearest safe zone: {telemetry.get('nearest_lz', 'unknown')}\n\n"
        f"Anomaly detected: {telemetry.get('anomaly', 'None')}. "
        f"What is your decision, Kairos Engine?"
    )
