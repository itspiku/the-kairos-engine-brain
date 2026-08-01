"""
The Kairos Engine - Gemma 4 E2B GGUF LLM Engine Interface

Production-grade wrapper with GPU auto-detection, multi-turn agentic loop,
token overflow protection, and retry with exponential backoff.
"""

import time
import json
import re
import logging
from typing import List, Dict, Any, Optional, Callable

from src.config import DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILENAME, MODEL_DIR, MAX_CONTEXT_TOKENS
from src.parser import GemmaParser
from src.core.exceptions import ModelLoadError

logger = logging.getLogger("kairos.llm")

START_TURN  = "<start_of_turn>"
END_TURN    = "<end_of_turn>"
START_MODEL = "<start_of_turn>model\n"

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_gemma_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    """Build the raw prompt string in Gemma 4 native format."""
    tool_block = ""
    if tools:
        tool_block = (
            "\n\nAvailable tools (call using <|tool_call|>call:name{key:<|\"|>val<|\"|>}<tool_call|>):\n"
            + json.dumps(tools, indent=2)
        )

    prompt = ""
    system_injected = False
    system_text = ""

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

        elif role == "tool_result":
            prompt += f"{START_TURN}user\n"
            prompt += f"[TOOL RESULT for {msg.get('tool_name', 'unknown')}]\n"
            prompt += f"{json.dumps(content, indent=1, default=str)}\n"
            prompt += f"[/TOOL RESULT]\n"
            prompt += f"Now reason about the tool result and decide your next action.{END_TURN}\n"

    prompt += START_MODEL
    return prompt


def _estimate_token_count(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def _detect_gpu_layers() -> int:
    """Auto-detect GPU availability and return n_gpu_layers setting."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — offloading all layers to GPU")
            return -1
    except ImportError:
        pass
    try:
        from llama_cpp import Llama
        # Try a quick check for CUDA support in llama-cpp-python build
        test = Llama.__init__.__doc__ or ""
        if "cublas" in test.lower() or "cuda" in test.lower():
            return -1
    except Exception:
        pass
    logger.info("No GPU detected — running on CPU")
    return 0


class GemmaLLMEngine:
    """Gemma 4 E2B GGUF inference engine with agentic multi-turn capabilities."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_batch: int = 256,
                 n_threads: int = 8, n_gpu_layers: Optional[int] = None):
        self.model_path = model_path
        self.n_ctx = n_ctx

        if n_gpu_layers is None:
            n_gpu_layers = _detect_gpu_layers()

        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                verbose=False,
            )
            logger.info(f"Gemma 4 loaded: {model_path} (ctx={n_ctx}, gpu_layers={n_gpu_layers})")
        except Exception as exc:
            raise ModelLoadError(model_path, reason=str(exc)) from exc

    # ── single-turn (backwards compatible) ────────────────────────────────
    def generate(self, messages: List[Dict], tools: List[Dict],
                 max_tokens: int = 512) -> Dict[str, Any]:
        """Single-turn generation — backwards compatible with original API."""
        prompt = _build_gemma_prompt(messages, tools)

        # Overflow protection: truncate if prompt is too long
        est_tokens = _estimate_token_count(prompt)
        if est_tokens > self.n_ctx - max_tokens:
            # Keep system + last user message, trim middle
            overflow = est_tokens - (self.n_ctx - max_tokens)
            trim_chars = overflow * 4
            prompt = prompt[:500] + "\n...[context trimmed]...\n" + prompt[500 + trim_chars:]
            logger.warning(f"Context overflow: trimmed ~{overflow} tokens")

        raw_text = self._call_llm(prompt, max_tokens)
        return self._parse_response(raw_text)

    # ── multi-turn agentic loop ───────────────────────────────────────────
    def agentic_generate(self, messages: List[Dict], tools: List[Dict],
                         tool_executor: Callable, max_tokens: int = 512,
                         max_iterations: int = 3) -> Dict[str, Any]:
        """
        Multi-turn agentic loop:
        1. LLM generates response (may contain tool calls)
        2. Tool calls are executed
        3. Results injected back into conversation
        4. LLM reasons again with tool results
        Repeats up to max_iterations times.
        """
        conversation = list(messages)
        all_tool_calls = []
        all_tool_results = []
        total_latency = 0.0

        for iteration in range(max_iterations):
            prompt = _build_gemma_prompt(conversation, tools)

            # Overflow protection
            est = _estimate_token_count(prompt)
            if est > self.n_ctx - max_tokens:
                overflow = est - (self.n_ctx - max_tokens)
                trim_chars = overflow * 4
                prompt = prompt[:500] + "\n...[trimmed]...\n" + prompt[500 + trim_chars:]

            raw_text = self._call_llm(prompt, max_tokens)
            parsed = self._parse_response(raw_text)
            total_latency += parsed["latency_sec"]

            if not parsed["tool_calls"]:
                # No tool calls — LLM has given final answer
                parsed["all_tool_calls"] = all_tool_calls
                parsed["all_tool_results"] = all_tool_results
                parsed["latency_sec"] = round(total_latency, 3)
                parsed["iterations"] = iteration + 1
                return parsed

            # Execute tool calls and inject results
            conversation.append({"role": "assistant", "content": raw_text})

            for tc in parsed["tool_calls"]:
                all_tool_calls.append(tc)
                try:
                    result = tool_executor(tc["name"], tc["arguments"])
                except Exception as exc:
                    result = {"error": str(exc)}
                all_tool_results.append({"name": tc["name"], "result": result})

                conversation.append({
                    "role": "tool_result",
                    "tool_name": tc["name"],
                    "content": result,
                })

            logger.info(f"Agentic iteration {iteration+1}: executed {len(parsed['tool_calls'])} tools")

        # Exceeded max iterations — return last response
        parsed["all_tool_calls"] = all_tool_calls
        parsed["all_tool_results"] = all_tool_results
        parsed["latency_sec"] = round(total_latency, 3)
        parsed["iterations"] = max_iterations
        return parsed

    # ── internal helpers ──────────────────────────────────────────────────
    def _call_llm(self, prompt: str, max_tokens: int, retries: int = 3) -> str:
        """Call the LLM with retry and exponential backoff."""
        last_error = None
        for attempt in range(retries):
            try:
                start = time.time()
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    stop=[END_TURN, "<eos>", "<end_of_turn>"],
                    echo=False,
                )
                latency = round(time.time() - start, 3)
                raw = output["choices"][0]["text"]
                # Stash latency for the caller
                self._last_latency = latency
                return raw
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {exc}. Retrying in {wait}s...")
                time.sleep(wait)
        raise ModelLoadError(self.model_path, reason=f"All {retries} retries failed: {last_error}")

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse raw LLM output into structured response."""
        tool_calls = GemmaParser.extract_tool_calls(raw_text)

        thinking = ""
        content = raw_text
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\|>|$)", raw_text, re.DOTALL)
        if not m:
            m = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            content = (raw_text[:m.start()] + raw_text[m.end():]).strip()

        latency = getattr(self, "_last_latency", 0.0)

        return {
            "raw": raw_text,
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls,
            "latency_sec": latency,
        }
