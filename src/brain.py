"""
Khumbu Engine - Gemma 4 Cognitive Autopilot Core Brain
"""

import time
from typing import List, Dict, Any
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
import torch

from src.parser import GemmaParser
from src.tools.definitions import get_tool_definitions
from src.tools.executor import ToolExecutor

class GemmaBrain:
    MODEL_ID = "google/gemma-4-E2B-it"
    
    def __init__(self):
        print("🧠 Initializing Khumbu Engine Gemma Brain...")
        print("   Loading Gemma-4-E2B with 4-bit quantization...")
        
        # 4-bit quantization config — mandatory for RTX 4050
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0},  # Force loading onto GPU 0
            attn_implementation="sdpa",  # faster than eager
            torch_dtype=torch.bfloat16,
        )
        
        self.tools = get_tool_definitions()
        self.executor = ToolExecutor()
        
        # Disable thinking mode for sub-2s latency (enable for complex decisions)
        self.thinking_mode = False
        
        print(f"   ✅ Model loaded. Device map: {self.model.hf_device_map}")
        print(f"   🎯 Ready for cognitive autopilot operations.\n")

    def generate(self, messages: List[Dict], max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate a response from Gemma 4 with tool-calling support.
        Returns parsed result with optional tool_calls.
        """
        start_time = time.time()
        
        # Apply chat template with tools
        inputs = self.processor.apply_chat_template(
            messages,
            tools=self.tools,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self.thinking_mode,
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with conservative settings for speed on RTX 4050
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic for safety-critical decisions
                temperature=None,
                top_p=None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_len:]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=False)
        
        # Parse thinking + content + tool_calls
        parsed = self.processor.parse_response(raw_output) if hasattr(self.processor, 'parse_response') else {"content": raw_output}
        
        # Extract any native tool calls
        tool_calls = GemmaParser.extract_tool_calls(raw_output)
        
        latency = time.time() - start_time
        
        return {
            "raw": raw_output,
            "content": parsed.get("content", raw_output),
            "thinking": parsed.get("thinking", ""),
            "tool_calls": tool_calls,
            "latency_sec": round(latency, 3)
        }

    def pre_flight_plan(self, mission_request: str, drone_id: str = "KHUMBU-01") -> Dict:
        """
        Phase A: Parse mission, gather intelligence, compute energy-aware route.
        """
        print("═" * 60)
        print("🛫 PHASE A: PRE-FLIGHT COGNITIVE PLANNING")
        print("═" * 60)
        
        system_prompt = (
            "You are the Khumbu Engine, an AI cognitive autopilot for BVLOS medical drones "
            "operating in the Himalayas. You make life-critical routing decisions by orchestrating "
            "weather, terrain, energy, and airspace tools. Be concise. Always think step-by-step."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mission_request}
        ]
        
        # Step 1: Gemma decides which intelligence tools to call
        print("\n[1] Gemma analyzing mission and selecting intelligence tools...")
        response = self.generate(messages, max_new_tokens=1024)
        
        print(f"   ⏱️  Latency: {response['latency_sec']}s")
        if response['thinking']:
            print(f"   💭 Thinking: {response['thinking'][:300]}...")
        
        # Execute tool calls if any
        tool_results = []
        if response['tool_calls']:
            print(f"   🔧 Tool calls detected: {[tc['name'] for tc in response['tool_calls']]}")
            for tc in response['tool_calls']:
                result = self.executor.execute(tc['name'], tc['arguments'])
                tool_results.append({
                    "name": tc['name'],
                    "arguments": tc['arguments'],
                    "result": result
                })
                print(f"      → {tc['name']}({tc['arguments']}) = {str(result)[:100]}")
        else:
            print("   ⚠️  No tool calls — Gemma responded directly.")
        
        # Step 2: Feed tool results back for final route decision
        if tool_results:
            messages.append({
                "role": "assistant",
                "tool_calls": [{"function": tc} for tc in response['tool_calls']],
                "tool_responses": [
                    {"name": tr['name'], "response": tr['result']} 
                    for tr in tool_results
                ]
            })
            
            print("\n[2] Gemma synthesizing final flight plan...")
            final = self.generate(messages, max_new_tokens=512)
            print(f"   ⏱️  Latency: {final['latency_sec']}s")
        else:
            final = response
        
        plan = {
            "phase": "pre_flight",
            "mission": mission_request,
            "tool_calls_made": len(response.get('tool_calls', [])),
            "tool_results": tool_results,
            "final_plan": final['content'],
            "total_latency_sec": round(sum([response['latency_sec'], final.get('latency_sec', 0)]), 3)
        }
        
        print(f"\n✅ Pre-flight plan complete. Total latency: {plan['total_latency_sec']}s")
        return plan

    def in_flight_decision(self, telemetry: Dict, mission_context: Dict) -> Dict:
        """
        Phase B: Called every 5 seconds (or on anomaly interrupt).
        """
        print("\n" + "═" * 60)
        print("🚁 PHASE B: IN-FLIGHT DYNAMIC RE-PLANNING")
        print("═" * 60)
        
        # Format telemetry into natural language for Gemma
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
                "role": "system", 
                "content": (
                    "You are the Khumbu Engine cognitive autopilot. A mid-flight anomaly has occurred. "
                    "You must decide: CONTINUE (stay on route), DIVERT (to contingency LZ), "
                    "LAND (immediate emergency landing), RE-TASK (handoff to another drone), or ABORT. "
                    "Use tools to verify before deciding. Respond with your decision and rationale."
                )
            },
            {"role": "user", "content": telemetry_text}
        ]
        
        print(f"\n[1] Anomaly: {telemetry.get('anomaly', 'routine check')}")
        print(f"    Position: {telemetry['lat']:.4f}, {telemetry['lon']:.4f} | Battery: {telemetry['battery_pct']}%")
        
        # For speed on RTX 4050, disable thinking in-flight
        prev_thinking = self.thinking_mode
        self.thinking_mode = False
        
        response = self.generate(messages, max_new_tokens=512)
        self.thinking_mode = prev_thinking  # restore
        
        print(f"   ⏱️  Decision latency: {response['latency_sec']}s")
        
        # Extract decision from content
        decision_text = response['content']
        
        # Parse explicit decision
        decision = "CONTINUE"  # default
        for d in ["DIVERT", "LAND", "ABORT", "RE-TASK", "CONTINUE"]:
            if d in decision_text.upper():
                decision = d
                break
        
        # Execute any follow-up tool calls
        tool_results = []
        if response['tool_calls']:
            for tc in response['tool_calls']:
                result = self.executor.execute(tc['name'], tc['arguments'])
                tool_results.append({"name": tc['name'], "result": result})
        
        # Log the decision
        self.executor.execute("log_decision", {
            "decision": decision,
            "rationale": decision_text[:500],
            "tool_calls": response.get('tool_calls', [])
        })
        
        result = {
            "phase": "in_flight",
            "decision": decision,
            "rationale": decision_text,
            "tool_calls": response.get('tool_calls', []),
            "tool_results": tool_results,
            "latency_sec": response['latency_sec'],
            "meets_sla": response['latency_sec'] < 2.0
        }
        
        print(f"   🎯 DECISION: {decision}")
        print(f"   {'✅' if result['meets_sla'] else '⚠️'} SLA: <2s requirement")
        
        return result
