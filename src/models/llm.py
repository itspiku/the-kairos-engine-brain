"""
The Kairos Engine - Gemma 4 E2B GGUF LLM Engine Interface
"""

import time
import json
import re
from typing import List, Dict, Any
from llama_cpp import Llama

from src.config import DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILENAME, MODEL_DIR
from src.parser import GemmaParser

START_TURN  = "<start_of_turn>"
END_TURN    = "<end_of_turn>"
START_MODEL = "<start_of_turn>model\n"


def _build_gemma_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    tool_block = ""
    if tools:
        tool_block = (
            "\n\nAvailable tools (call using <|tool_call|>call:name{key:<|\"|>val<|\"|>}<tool_call|>):\n"
            + json.dumps(tools, indent=2)
        )

    prompt = ""
    system_injected = False

    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            system_text = content + tool_block
            system_injected = True
            continue

        if role == "user":
            prompt += f"{START_TURN}user\n"
            if system_injected:
                prompt += f"[SYSTEM]\n{system_text}\n[/SYSTEM]\n\n"
                system_injected = False
            prompt += f"{content}{END_TURN}\n"

        elif role == "assistant":
            prompt += f"{START_TURN}model\n{content}{END_TURN}\n"

    prompt += START_MODEL
    return prompt


class GemmaLLMEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=4096,
            n_batch=256,
            n_threads=8,
            verbose=False
        )

    def generate(self, messages: List[Dict], tools: List[Dict], max_tokens: int = 512) -> Dict[str, Any]:
        prompt = _build_gemma_prompt(messages, tools)
        start = time.time()
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=[END_TURN, "<eos>", "<end_of_turn>"],
            echo=False
        )
        latency = round(time.time() - start, 3)

        raw_text = output["choices"][0]["text"]
        tool_calls = GemmaParser.extract_tool_calls(raw_text)

        thinking = ""
        content = raw_text
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\|>|$)", raw_text, re.DOTALL)
        if not m:
            m = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)

        if m:
            thinking = m.group(1).strip()
            content = (raw_text[:m.start()] + raw_text[m.end():]).strip()

        return {
            "raw": raw_text,
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls,
            "latency_sec": latency
        }
