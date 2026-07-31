"""
The Kairos Engine - Master Cognitive Autopilot Orchestrator
"""

import time
from typing import List, Dict, Any
from src.config import MAX_LATENCY_SLA_SEC, ENGINE_NAME
from src.models.llm import GemmaLLMEngine
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor
from src.utils.logger import KairosLogger
from src.utils.formatter import format_telemetry_prompt


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
        
        KairosLogger.ok("Kairos Master Orchestrator initialized & operational.\n")

    def pre_flight_plan(self, mission_request: str, drone_id: str = "KAIROS-01") -> Dict[str, Any]:
        KairosLogger.header("PHASE A: PRE-FLIGHT COGNITIVE PLANNING")
        
        system_prompt = (
            "You are The Kairos Engine, an AI cognitive autopilot for BVLOS medical drones "
            "operating in the Himalayas. You make life-critical routing decisions by orchestrating "
            "weather, terrain, energy, risk, and airspace tools. Be concise. Always think step-by-step."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mission_request}
        ]

        KairosLogger.info("Gemma 4 evaluating mission parameters & tool requirements...")
        res = self.llm_engine.generate(messages, self.tools, max_tokens=1024)

        KairosLogger.info(f"Gemma 4 Latency: {res['latency_sec']}s")

        tool_results = []
        if res["tool_calls"]:
            KairosLogger.info(f"Executing tool calls: {[tc['name'] for tc in res['tool_calls']]}")
            for tc in res["tool_calls"]:
                out = self.executor.execute(tc["name"], tc["arguments"])
                tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": out})
        else:
            KairosLogger.warn("Direct response generated without secondary tool invocation.")

        return {
            "phase": "pre_flight",
            "drone_id": drone_id,
            "mission": mission_request,
            "tool_calls": res.get("tool_calls", []),
            "tool_results": tool_results,
            "final_plan": res["content"],
            "total_latency_sec": res["latency_sec"]
        }

    def in_flight_decision(self, telemetry: Dict[str, Any], mission_context: Dict[str, Any] = None) -> Dict[str, Any]:
        KairosLogger.header("PHASE B: IN-FLIGHT DYNAMIC RE-PLANNING")
        
        telemetry_text = format_telemetry_prompt(telemetry)
        
        system_prompt = (
            "You are The Kairos Engine cognitive autopilot. A mid-flight anomaly has occurred. "
            "You must decide: CONTINUE (stay on route), DIVERT (to contingency LZ), "
            "LAND (immediate emergency landing), RE-TASK (handoff to another drone), or ABORT. "
            "Use assess_risk tool to verify before deciding. Respond with decision and rationale."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": telemetry_text}
        ]

        KairosLogger.warn(f"Anomaly detected: {telemetry.get('anomaly', 'routine check')}")
        res = self.llm_engine.generate(messages, self.tools, max_tokens=512)

        decision_text = res["content"]
        decision = "CONTINUE"
        for d in ["DIVERT", "LAND", "ABORT", "RE-TASK", "CONTINUE"]:
            if d in decision_text.upper():
                decision = d
                break

        meets_sla = res["latency_sec"] <= MAX_LATENCY_SLA_SEC
        KairosLogger.decision(decision, meets_sla=meets_sla)

        return {
            "phase": "in_flight",
            "decision": decision,
            "rationale": decision_text,
            "tool_calls": res.get("tool_calls", []),
            "latency_sec": res["latency_sec"],
            "meets_sla": meets_sla
        }
