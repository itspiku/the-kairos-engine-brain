"""
The Kairos Engine - Master Cognitive Autopilot Orchestrator

Combines Gemma 4 GGUF LLM reasoning with XGBoost risk prediction and
real-time anomaly detection for BVLOS medical drone operations.
Features multi-turn agentic tool execution, anomaly-aware decision making,
and full audit trail logging.
"""

import time
from typing import List, Dict, Any, Optional, Union

from src.config import MAX_LATENCY_SLA_SEC, ENGINE_NAME
from src.models.llm import GemmaLLMEngine
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor
from src.utils.logger import KairosLogger
from src.utils.formatter import format_telemetry_prompt, format_mission_summary
from src.core.telemetry import TelemetryReport
from src.core.types import FlightAction, CognitiveResponse
from src.core.exceptions import KairosError, SLAViolationError
from src.telemetry.anomaly_detector import KairosAnomalyDetector


class KairosEngine:
    """Master Cognitive Autopilot Orchestrator combining Gemma 4 GGUF & XGBoost Risk Prediction."""

    def __init__(self, model_path: str):
        KairosLogger.header("COGNITIVE AUTOPILOT INITIALIZATION")
        KairosLogger.info(f"Engine      : {ENGINE_NAME}")
        KairosLogger.info(f"LLM Core    : Gemma 4 E2B GGUF (4-bit Q4_K_M)")
        KairosLogger.info(f"Risk Model  : XGBoost Flight Risk Predictor")

        self.llm_engine = GemmaLLMEngine(model_path=model_path)
        self.tools = get_tool_definitions()
        self.executor = ToolExecutor()
        self.anomaly_detector = KairosAnomalyDetector()
        self._mission_log = []

        KairosLogger.ok("Kairos Master Orchestrator initialized & operational.\n")
        KairosLogger.audit("engine_initialized", {"model_path": model_path})

    def _to_telemetry(self, data: Union[Dict[str, Any], TelemetryReport]) -> TelemetryReport:
        """Convert raw dict to TelemetryReport if needed."""
        if isinstance(data, TelemetryReport):
            return data
        return TelemetryReport(
            lat=data.get("lat", 0.0),
            lon=data.get("lon", 0.0),
            altitude_m=data.get("altitude_m", 0.0),
            battery_pct=data.get("battery_pct", 100.0),
            drain_rate=data.get("drain_rate", 0.0),
            wind_speed=data.get("wind_speed", 0.0),
            wind_dir=data.get("wind_dir", 0.0),
            temp_c=data.get("temp_c", 15.0),
            payload_type=data.get("payload_type", "medical"),
            eta_min=data.get("eta_min"),
            nearest_lz=data.get("nearest_lz"),
            anomaly=data.get("anomaly"),
            priority=data.get("priority", "standard"),
        )

    def pre_flight_plan(self, mission_request: str, drone_id: str = "KAIROS-01") -> Dict[str, Any]:
        """Phase A: Pre-flight cognitive planning with multi-turn agentic tool execution."""
        KairosLogger.header("PHASE A: PRE-FLIGHT COGNITIVE PLANNING")

        system_prompt = (
            "You are The Kairos Engine, an AI cognitive autopilot for BVLOS medical drones "
            "operating in the Himalayas. You make life-critical routing decisions by orchestrating "
            "weather, terrain, energy, risk, and airspace tools. Be concise. Always think step-by-step. "
            "Use the available tools to gather data before making your final flight plan."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mission_request},
        ]

        KairosLogger.info("Gemma 4 evaluating mission parameters & tool requirements...")

        # Try multi-turn agentic loop first
        try:
            res = self.llm_engine.agentic_generate(
                messages, self.tools,
                tool_executor=self.executor.execute,
                max_tokens=1024,
                max_iterations=3,
            )
            tool_results = res.get("all_tool_results", [])
            tool_calls = res.get("all_tool_calls", [])
            iterations = res.get("iterations", 1)
            KairosLogger.info(f"Agentic loop completed in {iterations} iteration(s)")
        except Exception as exc:
            KairosLogger.warn(f"Agentic loop failed ({exc}), falling back to single-turn")
            res = self.llm_engine.generate(messages, self.tools, max_tokens=1024)
            tool_calls = res.get("tool_calls", [])
            tool_results = []
            # Execute tool calls from single-turn
            if tool_calls:
                KairosLogger.info(f"Executing tool calls: {[tc['name'] for tc in tool_calls]}")
                for tc in tool_calls:
                    out = self.executor.execute(tc["name"], tc["arguments"])
                    tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": out})

        KairosLogger.info(f"Gemma 4 Latency: {res['latency_sec']}s")

        if not tool_calls:
            KairosLogger.warn("Direct response generated without tool invocation.")

        result = {
            "phase": "pre_flight",
            "drone_id": drone_id,
            "mission": mission_request,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_plan": res["content"],
            "total_latency_sec": res["latency_sec"],
        }

        self._mission_log.append(result)
        KairosLogger.audit("pre_flight_plan", {
            "drone_id": drone_id,
            "tools_called": len(tool_calls),
            "latency": res["latency_sec"],
        })

        return result

    def in_flight_decision(self, telemetry: Union[Dict[str, Any], TelemetryReport],
                           mission_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Phase B: In-flight dynamic re-planning with anomaly detection."""
        KairosLogger.header("PHASE B: IN-FLIGHT DYNAMIC RE-PLANNING")

        # Normalize telemetry
        report = self._to_telemetry(telemetry)
        telem_dict = report.to_dict()

        # Run anomaly detection
        anomalies = self.anomaly_detector.detect_anomalies(telem_dict)
        severity = self.anomaly_detector.get_severity(telem_dict)
        recommended_action = self.anomaly_detector.recommend_action(telem_dict)

        if anomalies:
            for a in anomalies:
                KairosLogger.warn(f"ANOMALY: {a}")
            KairosLogger.info(f"Anomaly severity: {severity} | Recommended: {recommended_action}")

        telemetry_text = format_telemetry_prompt(telem_dict)

        system_prompt = (
            "You are The Kairos Engine cognitive autopilot. A mid-flight anomaly has occurred. "
            "You must decide: CONTINUE (stay on route), DIVERT (to contingency LZ), "
            "LAND (immediate emergency landing), RE-TASK (handoff to another drone), or ABORT. "
            "Use assess_risk tool to verify before deciding. Respond with decision and rationale."
        )

        if anomalies:
            system_prompt += f"\n\nDetected anomalies: {'; '.join(anomalies)}"
            system_prompt += f"\nSeverity: {severity}. System recommends: {recommended_action}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": telemetry_text},
        ]

        KairosLogger.warn(f"Anomaly detected: {telem_dict.get('anomaly', 'routine check')}")

        # Use agentic loop for in-flight decisions too
        try:
            res = self.llm_engine.agentic_generate(
                messages, self.tools,
                tool_executor=self.executor.execute,
                max_tokens=512,
                max_iterations=2,
            )
        except Exception:
            res = self.llm_engine.generate(messages, self.tools, max_tokens=512)

        decision_text = res["content"]

        # Parse decision from LLM output
        decision = FlightAction.CONTINUE.value
        for d in [FlightAction.DIVERT, FlightAction.LAND, FlightAction.ABORT,
                  FlightAction.RE_TASK, FlightAction.RTL, FlightAction.CONTINUE]:
            if d.value in decision_text.upper():
                decision = d.value
                break

        meets_sla = res["latency_sec"] <= MAX_LATENCY_SLA_SEC
        KairosLogger.decision(decision, meets_sla=meets_sla)

        result = {
            "phase": "in_flight",
            "decision": decision,
            "rationale": decision_text,
            "anomalies": anomalies,
            "severity": severity,
            "recommended_action": recommended_action,
            "tool_calls": res.get("all_tool_calls", res.get("tool_calls", [])),
            "latency_sec": res["latency_sec"],
            "meets_sla": meets_sla,
        }

        self._mission_log.append(result)
        KairosLogger.audit("in_flight_decision", {
            "decision": decision,
            "severity": severity,
            "latency": res["latency_sec"],
            "meets_sla": meets_sla,
        })

        # Log decision via tool executor
        try:
            self.executor.execute("log_decision", {
                "decision": decision,
                "rationale": decision_text[:200],
                "tool_calls": res.get("all_tool_calls", []),
            })
        except Exception:
            pass

        return result

    def post_flight_report(self) -> str:
        """Generate a post-flight mission summary report."""
        KairosLogger.header("POST-FLIGHT MISSION REPORT")

        plan = next((r for r in self._mission_log if r.get("phase") == "pre_flight"), None)
        decision = next((r for r in self._mission_log if r.get("phase") == "in_flight"), None)

        summary = format_mission_summary(plan, decision)
        print(summary)

        KairosLogger.audit("post_flight_report", {
            "phases_completed": len(self._mission_log),
            "plan_latency": plan.get("total_latency_sec") if plan else None,
            "decision": decision.get("decision") if decision else None,
        })

        return summary

    def get_status(self) -> Dict[str, Any]:
        """Return current engine status."""
        return {
            "engine": ENGINE_NAME,
            "phases_completed": len(self._mission_log),
            "tools_available": len(self.tools),
            "last_decision": self._mission_log[-1].get("decision") if self._mission_log else None,
        }
