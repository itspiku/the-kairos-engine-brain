"""
The Kairos Engine - Global Configuration & Environment Settings
"""

import os
from pathlib import Path

# Engine Metadata
ENGINE_NAME = "The Kairos Engine"
ENGINE_VERSION = "2.4.0"
AUTHOR = "Kairos Autopilot Systems"

# Model Configuration
DEFAULT_MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_MODEL_FILENAME = "gemma-4-E2B-it-Q4_K_M.gguf"
MODEL_DIR = Path("./models")

# Risk Predictor Configuration
RISK_MODEL_PATH = "kairos_crash_predictor.json"
CRITICAL_RISK_THRESHOLD = 0.70
HIGH_RISK_THRESHOLD = 0.40

# Operational SLAs
MAX_LATENCY_SLA_SEC = 2.0  # Real-time sub-2s requirement for emergency re-planning
MAX_CONTEXT_TOKENS = 4096

# Geographical Default Bounds (Himalayan Flight Corridor: Pokhara to Muktinath)
DEFAULT_ORIGIN = [28.2096, 83.9856]      # Pokhara Base
DEFAULT_DESTINATION = [28.8167, 83.8667] # Muktinath Clinic
