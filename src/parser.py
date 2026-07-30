"""
Parser for Gemma 4's Native Function Calling DSL.
"""

import re
from typing import List, Dict, Any

class GemmaParser:
    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """
        Parse Gemma 4's NATIVE tool call format:
        <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
        
        This is NOT OpenAI-style JSON. Gemma uses its own DSL.
        Ref: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
        """
        def cast_value(v):
            v = v.strip()
            try: return int(v)
            except:
                try: return float(v)
                except:
                    low = v.lower()
                    if low == 'true': return True
                    if low == 'false': return False
                    return v.strip("'\"")

        pattern = r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        
        calls = []
        for name, args_str in matches:
            # Parse key:value pairs handling Gemma's <|"|> string delimiters
            arg_pattern = r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))'
            args = {}
            for k, v_quoted, v_raw in re.findall(arg_pattern, args_str):
                args[k] = cast_value(v_quoted if v_quoted else v_raw)
            calls.append({"name": name, "arguments": args})
        
        return calls
