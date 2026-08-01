"""
The Kairos Engine - Output Formatters & Payload Serialization Utilities

Provides formatters for telemetry prompts, tool results, mission summaries,
and risk reports used across the CLI, LLM prompts, and audit logs.
"""

import json
from typing import Dict, Any, Optional, List


def format_json_pretty(data: Dict[str, Any]) -> str:
    """Format dictionary into pretty JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


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


def format_tool_result_prompt(tool_name: str, tool_args: Dict[str, Any],
                               result: Any) -> str:
    """Format a tool execution result for injection into the LLM conversation."""
    args_str = ", ".join(f"{k}={v}" for k, v in (tool_args or {}).items())
    result_str = json.dumps(result, indent=1, default=str) if isinstance(result, dict) else str(result)
    return (
        f"Tool `{tool_name}({args_str})` returned:\n"
        f"{result_str}"
    )


def format_mission_summary(plan_result: Optional[Dict[str, Any]] = None,
                           decision_result: Optional[Dict[str, Any]] = None) -> str:
    """Format a complete post-flight mission summary report."""
    lines = [
        "=" * 64,
        "  🚁 THE KAIROS ENGINE — POST-FLIGHT MISSION SUMMARY",
        "=" * 64,
    ]

    if plan_result:
        lines.append(f"\n📋 PHASE A: PRE-FLIGHT PLAN")
        lines.append(f"   Drone ID     : {plan_result.get('drone_id', 'N/A')}")
        lines.append(f"   Mission      : {str(plan_result.get('mission', 'N/A'))[:80]}...")
        lines.append(f"   Tools Called  : {len(plan_result.get('tool_calls', []))}")
        lines.append(f"   Latency      : {plan_result.get('total_latency_sec', 'N/A')}s")

        tool_results = plan_result.get("tool_results", [])
        if tool_results:
            lines.append(f"   Tool Results  :")
            for tr in tool_results:
                lines.append(f"     • {tr.get('name', '?')}: {_summarize_result(tr.get('result', {}))}")

    if decision_result:
        lines.append(f"\n🎯 PHASE B: IN-FLIGHT DECISION")
        lines.append(f"   Decision     : {decision_result.get('decision', 'N/A')}")
        lines.append(f"   Latency      : {decision_result.get('latency_sec', 'N/A')}s")
        lines.append(f"   Meets SLA    : {'YES ✅' if decision_result.get('meets_sla', False) else 'NO ⚠️'}")
        rationale = str(decision_result.get("rationale", ""))[:200]
        if rationale:
            lines.append(f"   Rationale    : {rationale}")

    lines.append(f"\n{'=' * 64}")
    return "\n".join(lines)


def format_risk_report(risk_data: Dict[str, Any]) -> str:
    """Pretty-print a risk assessment result."""
    if not risk_data:
        return "   [RISK] No risk data available."

    prob = risk_data.get("crash_probability", 0)
    level = risk_data.get("risk_level", "Unknown")
    source = risk_data.get("data_source", "N/A")

    icon = {"Critical": "🔴", "High": "🟠", "Acceptable": "🟢"}.get(level, "⚪")

    lines = [
        f"   {icon} RISK ASSESSMENT",
        f"     Crash Probability : {prob:.0%}",
        f"     Risk Level        : {level}",
        f"     Data Source       : {source}",
    ]

    if "confidence_interval" in risk_data:
        ci = risk_data["confidence_interval"]
        lines.append(f"     95% CI            : [{ci[0]:.2%}, {ci[1]:.2%}]")

    if "feature_contributions" in risk_data:
        lines.append(f"     Top Factors       :")
        for feat, contrib in risk_data["feature_contributions"].items():
            lines.append(f"       • {feat}: {contrib:+.3f}")

    return "\n".join(lines)


def _summarize_result(result: Any) -> str:
    """Create a one-line summary of a tool result."""
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {result['error']}"
        keys = list(result.keys())[:3]
        parts = [f"{k}={result[k]}" for k in keys]
        return ", ".join(parts)
    if isinstance(result, list):
        return f"[{len(result)} items]"
    return str(result)[:100]
