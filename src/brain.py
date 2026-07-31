"""
The Kairos Engine - Cognitive Autopilot Core Brain Interface
"""

from typing import List, Dict, Any
from src.core.engine import KairosEngine


class GemmaBrain(KairosEngine):
    """
    GemmaBrain wrapper class maintaining full backwards compatibility
    with the Kairos Master Engine Orchestrator.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "./models/gemma-4-E2B-it-Q4_K_M.gguf"
        super().__init__(model_path=model_path)


class KairosBrain(KairosEngine):
    """Alias for KairosEngine."""
    pass
