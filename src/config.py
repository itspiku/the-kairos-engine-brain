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

# ---------------------------------------------------------------------------
# Airframe Energy Profile
# ---------------------------------------------------------------------------
# Reference platform: heavy-lift VTOL/fixed-wing hybrid medical delivery UAV.
#   MTOW ~25 kg, ~3 kg medical payload, 1800 Wh pack (~12 kg at 150 Wh/kg).
#
# These replace a set of unsourced constants that were mutually inconsistent:
# 25 Wh/km against an 800 Wh pack implied a 32 km range on a 68.5 km corridor,
# so every mission computed as infeasible even at a full charge.
#
# Each value below is derived rather than picked:
#   ENERGY_PER_KM_WH      ~650 W cruise at 20 m/s (72 km/h) -> 9.0 Wh/km
#   ENERGY_PER_CLIMB_M_WH mgh for 25 kg = 0.068 Wh/m, at ~65% climb efficiency
#   CRUISE_SPEED_MS       20 m/s, the speed the above power figure assumes
#
# Corridor sanity check (Pokhara -> Muktinath, 68.5 km, ~3,700 m of climb):
#   distance 616 Wh + climb 370 Wh + wind ~222 Wh at 12 m/s = ~1,208 Wh,
#   which is ~67% of the pack and leaves a workable reserve.
BATTERY_CAPACITY_WH   = 1800.0  # Usable pack energy
CRUISE_SPEED_MS       = 20.0    # Nominal cruise airspeed
ENERGY_PER_KM_WH      = 9.0     # Level-flight cruise burn
ENERGY_PER_CLIMB_M_WH = 0.10    # Additional burn per metre of climb
ENERGY_PER_KM_PER_DEG = 0.02    # Additional burn per km per °C below 15
WIND_ADVERSE_FRACTION = 0.6     # Share of wind speed treated as headwind.
                                # Placeholder until per-edge bearing lands and
                                # head/cross/tail components are resolved
                                # properly; the router has no wind direction yet.
ENERGY_RESERVE_FRACTION = 0.15  # Required margin before a route counts as feasible
