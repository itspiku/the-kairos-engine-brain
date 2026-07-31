"""
Khumbu Engine - Gemma 4 E2B Cognitive Autopilot Core Brain
Backend: llama-cpp-python (GGUF / C++ runtime)
Model  : gemma-4-E2B-it-Q4_K_M.gguf  (~3.46 GB)
"""

import json
import time
from typing import List, Dict, Any

from llama_cpp import Llama

from src.parser import GemmaParser
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor


# ── Gemma 4 chat tokens ───────────────────────────────────────────────────────
# NOTE: llama-cpp-python adds <bos> automatically — do NOT prepend it manually
START_TURN  = "<start_of_turn>"
END_TURN    = "<end_of_turn>"
START_MODEL = "<start_of_turn>model\n"


def _build_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    """
    Render a Gemma-4 instruct prompt string.
    System message is prepended to the first user turn since Gemma 4
    handles system context via the user turn.
    Tool schemas are appended to the system context so the model knows
    what functions it can call.
    """
    # Serialise tool definitions so Gemma knows how to call them
    tool_block = ""
    if tools:
        tool_block = (
            "\n\nAvailable tools (call using <|tool_call|>call:name{key:<|\"|>val<|\"|>}<tool_call|>):\n"
            + json.dumps(tools, indent=2)
        )

    prompt = ""   # llama-cpp-python prepends <bos> automatically
    system_injected = False

    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            # We'll inject this into the first user turn
            system_text = content + tool_block
            system_injected = True
            continue  # defer

        if role == "user":
            prompt += f"{START_TURN}user\n"
            if system_injected:
                prompt += f"[SYSTEM]\n{system_text}\n[/SYSTEM]\n\n"
                system_injected = False
            prompt += f"{content}{END_TURN}\n"

        elif role == "assistant":
            # Re-inject previous model turn (for multi-turn)
            prompt += f"{START_TURN}model\n{content}{END_TURN}\n"

    # Open the model generation slot
    prompt += START_MODEL
    return prompt


class GemmaBrain:
    REPO_ID   = "unsloth/gemma-4-E2B-it-GGUF"
    FILENAME  = "gemma-4-E2B-it-Q4_K_M.gguf"
    MODEL_DIR = "./models"

    def __init__(self, model_path: str | None = None):
        print("[BRAIN] Initializing Khumbu Engine - Gemma 4 E2B (GGUF / C++ runtime)")
        print(f"   Model : {self.FILENAME}")
        print(f"   Quant : Q4_K_M  (4-bit, ~3.46 GB)")

        path = model_path or f"{self.MODEL_DIR}/{self.FILENAME}"

        print(f"   Loading from : {path}")
        print("   Backend      : llama-cpp-python (CPU mode — no CUDA Toolkit installed)\n")

        self.llm = Llama(
            model_path    = path,
            n_gpu_layers  = 0,        # CPU-only (CUDA Toolkit not installed)
            n_ctx         = 4096,     # reduced ctx for CPU speed
            n_batch       = 256,
            n_threads     = 8,        # use all physical cores
            verbose       = False,
        )

        self.tools    = get_tool_definitions()
        self.executor = ToolExecutor()

        print("   [OK] Model loaded and ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    def generate(self, messages: List[Dict], max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate a response using the GGUF model.
        Returns a dict with keys: raw, content, thinking, tool_calls, latency_sec.
        """
        prompt = _build_prompt(messages, self.tools)

        start = time.time()
        output = self.llm(
            prompt,
            max_tokens    = max_new_tokens,
            temperature   = 0.0,          # deterministic for safety-critical ops
            top_p         = 1.0,
            stop          = [END_TURN, "<eos>", "<end_of_turn>"],
            echo          = False,
        )
        latency = round(time.time() - start, 3)

        raw_text   = output["choices"][0]["text"]
        tool_calls = GemmaParser.extract_tool_calls(raw_text)

        # Strip thinking tags if present
        thinking = ""
        content  = raw_text
        import re
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\|>|$)", raw_text, re.DOTALL)
        if not m:
            m = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)

        if m:
            thinking = m.group(1).strip()
            content  = (raw_text[:m.start()] + raw_text[m.end():]).strip()

        return {
            "raw"        : raw_text,
            "content"    : content,
            "thinking"   : thinking,
            "tool_calls" : tool_calls,
            "latency_sec": latency,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def pre_flight_plan(self, mission_request: str, drone_id: str = "KHUMBU-01") -> Dict:
        """
        Phase A: Parse mission, gather intelligence, compute energy-aware route.
        """
        print("=" * 60)
        print("[PHASE-A] PRE-FLIGHT COGNITIVE PLANNING")
        print("=" * 60)

        system_prompt = (
            "You are the Khumbu Engine, an AI cognitive autopilot for BVLOS medical drones "
            "operating in the Himalayas. You make life-critical routing decisions by orchestrating "
            "weather, terrain, energy, and airspace tools. Be concise. Always think step-by-step."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": mission_request},
        ]

        # Step 1: Gemma decides which intelligence tools to call
        print("\n[1] Gemma analyzing mission and selecting intelligence tools...")
        response = self.generate(messages, max_new_tokens=1024)

        print(f"   ⏱️  Latency: {response['latency_sec']}s")
        if response["thinking"]:
            print(f"   💭 Thinking: {response['thinking'][:300]}...")

        # Execute tool calls if any
        tool_results = []
        if response["tool_calls"]:
            print(f"   [TOOLS] Tool calls: {[tc['name'] for tc in response['tool_calls']]}")
            for tc in response["tool_calls"]:
                result = self.executor.execute(tc["name"], tc["arguments"])
                tool_results.append({
                    "name"     : tc["name"],
                    "arguments": tc["arguments"],
                    "result"   : result,
                })
                print(f"      → {tc['name']}({tc['arguments']}) = {str(result)[:100]}")
        else:
            print("   [WARN] No tool calls - Gemma responded directly.")

        # Step 2: Feed tool results back for final route synthesis
        if tool_results:
            # Append assistant turn with tool call results as context
            tool_context = "\n".join(
                f"Tool `{tr['name']}` returned: {json.dumps(tr['result'])}"
                for tr in tool_results
            )
            messages.append({"role": "assistant", "content": response["content"]})
            messages.append({
                "role"   : "user",
                "content": f"Tool results:\n{tool_context}\n\nNow synthesize the final flight plan.",
            })

            print("\n[2] Gemma synthesizing final flight plan...")
            final = self.generate(messages, max_new_tokens=512)
            print(f"   ⏱️  Latency: {final['latency_sec']}s")
        else:
            final = response

        plan = {
            "phase"           : "pre_flight",
            "drone_id"        : drone_id,
            "mission"         : mission_request,
            "tool_calls_made" : len(response.get("tool_calls", [])),
            "tool_results"    : tool_results,
            "final_plan"      : final["content"],
            "total_latency_sec": round(
                response["latency_sec"] + final.get("latency_sec", 0), 3
            ),
        }

        print(f"\n[OK] Pre-flight plan complete. Total latency: {plan['total_latency_sec']}s")
        return plan

    # ──────────────────────────────────────────────────────────────────────────
    def in_flight_decision(self, telemetry: Dict, mission_context: Dict) -> Dict:
        """
        Phase B: Called every 5 seconds (or on anomaly interrupt).
        """
        print("\n" + "=" * 60)
        print("[PHASE-B] IN-FLIGHT DYNAMIC RE-PLANNING")
        print("=" * 60)

        telemetry_text = (
            f"TELEMETRY REPORT:\n"
            f"- Position: {telemetry['lat']:.4f}, {telemetry['lon']:.4f} @ {telemetry['altitude_m']}m\n"
            f"- Battery: {telemetry['battery_pct']}% (draining at {telemetry['drain_rate']}%/min)\n"
            f"- Wind: {telemetry['wind_speed']} m/s from {telemetry['wind_dir']}°\n"
            f"- Temperature: {telemetry['temp_c']}°C\n"
            f"- Payload: {telemetry['payload_type']} (Priority: {telemetry.get('priority', 'standard')})\n"
            f"- Time to destination: {telemetry.get('eta_min', '?')} min\n"
            f"- Nearest safe zone: {telemetry.get('nearest_lz', 'unknown')}\n\n"
            f"Anomaly detected: {telemetry.get('anomaly', 'None')}. "
            f"What is your decision, Khumbu Engine?"
        )

        messages = [
            {
                "role"   : "system",
                "content": (
                    "You are the Khumbu Engine cognitive autopilot. A mid-flight anomaly has occurred. "
                    "You must decide: CONTINUE (stay on route), DIVERT (to contingency LZ), "
                    "LAND (immediate emergency landing), RE-TASK (handoff to another drone), or ABORT. "
                    "Use tools to verify before deciding. Respond with your decision and rationale."
                ),
            },
            {"role": "user", "content": telemetry_text},
        ]

        print(f"\n[1] Anomaly: {telemetry.get('anomaly', 'routine check')}")
        print(f"    Position: {telemetry['lat']:.4f}, {telemetry['lon']:.4f} | Battery: {telemetry['battery_pct']}%")

        response = self.generate(messages, max_new_tokens=512)
        print(f"   ⏱️  Decision latency: {response['latency_sec']}s")

        decision_text = response["content"]

        # Parse explicit decision keyword
        decision = "CONTINUE"  # default
        for d in ["DIVERT", "LAND", "ABORT", "RE-TASK", "CONTINUE"]:
            if d in decision_text.upper():
                decision = d
                break

        # Execute any follow-up tool calls
        tool_results = []
        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                result = self.executor.execute(tc["name"], tc["arguments"])
                tool_results.append({"name": tc["name"], "result": result})

        # Audit log the decision
        self.executor.execute("log_decision", {
            "decision"  : decision,
            "rationale" : decision_text[:500],
            "tool_calls": response.get("tool_calls", []),
        })

        result = {
            "phase"      : "in_flight",
            "decision"   : decision,
            "rationale"  : decision_text,
            "tool_calls" : response.get("tool_calls", []),
            "tool_results": tool_results,
            "latency_sec": response["latency_sec"],
            "meets_sla"  : response["latency_sec"] < 2.0,
        }

        print(f"   [DECISION]: {decision}")
        print(f"   {'[SLA-OK]' if result['meets_sla'] else '[SLA-FAIL]'} SLA: <2s requirement")
        return result
