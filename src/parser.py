"""
Parser for Gemma 4's Native Function Calling DSL.

Supports:
- Gemma 4 native DSL: <|tool_call|>call:func{key:<|"|>val<|"|>}<tool_call|>
- Nested objects: {key:{nested_key:val}}
- Array values: [val1,val2,[nested]]
- JSON fallback: {"name":"func","arguments":{...}}
- Input validation against tool schemas
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from src.core.exceptions import ParsingError

logger = logging.getLogger("kairos.parser")


class GemmaParser:
    """Parses Gemma 4 LLM output into structured tool calls."""

    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM output.
        Tries Gemma 4 native DSL first, then falls back to JSON parsing.
        Returns list of {"name": str, "arguments": dict}.
        """
        if not text or not text.strip():
            return []

        # Attempt 1: Gemma 4 native DSL
        calls = GemmaParser._parse_native_dsl(text)
        if calls:
            return calls

        # Attempt 2: JSON fallback
        calls = GemmaParser._parse_json_fallback(text)
        if calls:
            return calls

        return []

    # Opening delimiter varies by how the model closes the tag. Verified against
    # real Gemma output, which emits the full `<|tool_call|>call:` form:
    #   <|tool_call|>call:name{...}<tool_call|>   <- what the model actually produces
    #   <|tool_call|name{...}<tool_call|>         <- abbreviated form
    #   <|tool_call>name{...}<tool_call|>
    # `\|?>?` accepts `|`, `>`, `|>`, or neither; an earlier `(?:\||\>)?` matched
    # only one of the two characters and so silently dropped every real tool call.
    _DSL_PATTERN = r"<\|tool_call\|?>?\s*(?:call:)?(\w+)\s*\{(.*?)\}\s*<tool_call\|?>"

    @staticmethod
    def _parse_native_dsl(text: str) -> List[Dict[str, Any]]:
        """Parse Gemma 4 native tool call DSL format."""
        matches = re.findall(GemmaParser._DSL_PATTERN, text, re.DOTALL)

        calls = []
        for name, args_str in matches:
            try:
                args = GemmaParser._parse_dsl_arguments(args_str)
                calls.append({"name": name, "arguments": args})
            except Exception as exc:
                logger.warning(f"Failed to parse DSL args for '{name}': {exc}")
                calls.append({"name": name, "arguments": {}})
        return calls

    @staticmethod
    def _parse_dsl_arguments(args_str: str) -> Dict[str, Any]:
        """
        Parse key:value pairs from Gemma DSL, supporting:
        - String values with <|"|> delimiters
        - Nested objects: {key:val}
        - Arrays: [val1,val2]
        - Numeric, boolean literals
        """
        args = {}
        i = 0
        s = args_str.strip()

        while i < len(s):
            # Skip whitespace and commas
            while i < len(s) and s[i] in ' ,\n\t':
                i += 1
            if i >= len(s):
                break

            # Parse key
            key_start = i
            while i < len(s) and s[i] not in ':':
                i += 1
            key = s[key_start:i].strip()
            if not key:
                break
            i += 1  # skip ':'

            # Skip whitespace
            while i < len(s) and s[i] in ' \t':
                i += 1
            if i >= len(s):
                break

            # Parse value
            value, i = GemmaParser._parse_value(s, i)
            args[key] = value

        return args

    @staticmethod
    def _parse_value(s: str, i: int):
        """Parse a single value starting at position i. Returns (value, new_i)."""
        if i >= len(s):
            return "", i

        # Gemma string delimiter: <|"|>...<|"|>
        if s[i:].startswith('<|"|>'):
            i += 5  # skip <|"|>
            val_start = i
            end_delim = s.find('<|"|>', i)
            if end_delim == -1:
                val = s[val_start:]
                return val, len(s)
            val = s[val_start:end_delim]
            return val, end_delim + 5

        # Nested object
        if s[i] == '{':
            depth = 1
            start = i + 1
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                i += 1
            inner = s[start:i - 1]
            return GemmaParser._parse_dsl_arguments(inner), i

        # Array
        if s[i] == '[':
            depth = 1
            start = i + 1
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == '[':
                    depth += 1
                elif s[i] == ']':
                    depth -= 1
                i += 1
            inner = s[start:i - 1]
            return GemmaParser._parse_array(inner), i

        # Quoted string
        if s[i] in ('"', "'"):
            quote = s[i]
            i += 1
            val_start = i
            while i < len(s) and s[i] != quote:
                if s[i] == '\\':
                    i += 1
                i += 1
            val = s[val_start:i]
            if i < len(s):
                i += 1  # skip closing quote
            return val, i

        # Raw value (until comma, }, or end)
        val_start = i
        while i < len(s) and s[i] not in ',}]\n':
            i += 1
        raw = s[val_start:i].strip()
        return GemmaParser._cast_value(raw), i

    @staticmethod
    def _parse_array(inner: str) -> list:
        """Parse comma-separated array contents, supporting nested arrays/objects."""
        result = []
        i = 0
        s = inner.strip()
        while i < len(s):
            while i < len(s) and s[i] in ' ,\t\n':
                i += 1
            if i >= len(s):
                break
            val, i = GemmaParser._parse_value(s, i)
            result.append(val)
        return result

    @staticmethod
    def _cast_value(v):
        """Cast a raw string value to the appropriate Python type."""
        v = v.strip()
        if not v:
            return ""
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
        low = v.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        if low == 'null' or low == 'none':
            return None
        return v.strip("'\"")

    @staticmethod
    def _parse_json_fallback(text: str) -> List[Dict[str, Any]]:
        """
        Fallback: try to find JSON-formatted tool calls in the text.
        Looks for patterns like: {"name": "func_name", "arguments": {...}}
        """
        calls = []

        # Try to find JSON objects with "name" and "arguments" keys
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if "name" in parsed and "arguments" in parsed:
                    calls.append({
                        "name": parsed["name"],
                        "arguments": parsed["arguments"],
                    })
            except json.JSONDecodeError:
                continue

        # Also try parsing the entire text as a JSON array of tool calls
        if not calls:
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            calls.append({
                                "name": item["name"],
                                "arguments": item.get("arguments", {}),
                            })
                elif isinstance(parsed, dict) and "name" in parsed:
                    calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    })
            except (json.JSONDecodeError, TypeError):
                pass

        return calls

    @staticmethod
    def validate_tool_call(call: Dict[str, Any], tool_schemas: List[Dict]) -> List[str]:
        """
        Validate a parsed tool call against the tool definition schemas.
        Returns a list of validation error strings (empty = valid).
        """
        errors = []
        tool_name = call.get("name", "")
        arguments = call.get("arguments", {})

        # Find matching schema
        schema = None
        for tool_def in tool_schemas:
            func = tool_def.get("function", tool_def)
            if func.get("name") == tool_name:
                schema = func
                break

        if schema is None:
            errors.append(f"Unknown tool: '{tool_name}'")
            return errors

        params = schema.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        # Check required parameters
        for req in required:
            if req not in arguments:
                errors.append(f"Missing required parameter '{req}' for tool '{tool_name}'")

        # Check for unknown parameters
        for key in arguments:
            if key not in properties:
                errors.append(f"Unknown parameter '{key}' for tool '{tool_name}'")

        return errors
