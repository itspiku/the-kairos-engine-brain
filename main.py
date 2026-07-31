"""
Khumbu Engine — Entry point
Downloads Gemma 4 E2B Q4_K_M GGUF if not present, then runs the demo mission.
"""
import sys
import io
# Force UTF-8 on Windows stdout so Unicode characters display correctly
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
from pathlib import Path

# ── Model download ─────────────────────────────────────────────────────────────
REPO_ID   = "unsloth/gemma-4-E2B-it-GGUF"
FILENAME  = "gemma-4-E2B-it-Q4_K_M.gguf"
MODEL_DIR = Path("./models")

def ensure_model() -> str:
    """
    Download the GGUF model from Hugging Face if it is not already present.
    Returns the local path to the model file.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / FILENAME

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print(f"[OK] Model already present: {model_path}  ({size_gb:.2f} GB)")
        return str(model_path)

    print(f"[>>] Model not found. Downloading from HuggingFace...")
    print(f"   Repo     : {REPO_ID}")
    print(f"   File     : {FILENAME}")
    print(f"   Dest     : {model_path}")
    print(f"   Size     : ~3.46 GB — this may take a few minutes.\n")

    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id   = REPO_ID,
        filename  = FILENAME,
        local_dir = str(MODEL_DIR),
    )
    print(f"\n[OK] Download complete: {local}\n")
    return local


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Ensure the GGUF model is available
    model_path = ensure_model()

    # 2. Initialise the brain (loads GGUF into GPU via llama-cpp-python)
    from src.brain import GemmaBrain
    brain = GemmaBrain(model_path=model_path)

    # ── TEST 1: Pre-flight Planning ──
    mission = (
        "Send oxytocin to the clinic in Muktinath (28.8167, 83.8667) "
        "before the storm hits in 90 minutes. Origin is Pokhara (28.2096, 83.9856)."
    )
    plan = brain.pre_flight_plan(mission)

    print("\n[PLAN] FLIGHT PLAN SUMMARY:")
    print(json.dumps(plan, indent=2))

    # ── TEST 2: In-flight Anomaly ──
    telemetry = {
        "lat": 28.3500, "lon": 83.8800, "altitude_m": 3200,
        "battery_pct": 42, "drain_rate": 4.2,
        "wind_speed": 18, "wind_dir": 270,
        "temp_c": -5, "payload_type": "oxytocin",
        "eta_min": 25, "nearest_lz": "LZ_03",
        "anomaly": "Battery drain rate doubled; headwind increased to 18 m/s",
    }

    decision = brain.in_flight_decision(telemetry, plan)

    print("\n[DECISION] FLIGHT DECISION SUMMARY:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
