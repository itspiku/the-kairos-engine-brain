"""
The Kairos Engine — CLI Entry Point
Downloads Gemma 4 E2B Q4_K_M GGUF if not present, then runs the demo mission.
"""

import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
from src.config import DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILENAME, MODEL_DIR, DEFAULT_ORIGIN, DEFAULT_DESTINATION


def ensure_model() -> str:
    """Download the GGUF model from Hugging Face if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / DEFAULT_MODEL_FILENAME

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print(f"[OK] Model present: {model_path} ({size_gb:.2f} GB)")
        return str(model_path)

    print(f"[>>] Downloading Gemma 4 GGUF from HuggingFace...")
    print(f"   Repo : {DEFAULT_MODEL_REPO}")
    print(f"   File : {DEFAULT_MODEL_FILENAME}")
    print(f"   Dest : {model_path}\n")

    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=DEFAULT_MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
    )
    print(f"\n[OK] Download complete: {local}\n")
    return local


def main():
    model_path = ensure_model()

    from src.brain import KairosBrain
    brain = KairosBrain(model_path=model_path)

    # ── MISSION A: Pre-flight Planning ──
    mission = (
        f"Send oxytocin to the clinic in Muktinath ({DEFAULT_DESTINATION[0]}, {DEFAULT_DESTINATION[1]}) "
        f"before the storm hits in 90 minutes. Origin is Pokhara Base ({DEFAULT_ORIGIN[0]}, {DEFAULT_ORIGIN[1]})."
    )
    plan = brain.pre_flight_plan(mission, drone_id="KAIROS-01")

    print("\n[PLAN] KAIROS FLIGHT PLAN SUMMARY:")
    print(json.dumps(plan, indent=2))

    # ── MISSION B: In-flight Anomaly Re-planning ──
    telemetry = {
        "lat": 28.3500, "lon": 83.8800, "altitude_m": 3200,
        "battery_pct": 42, "drain_rate": 4.2,
        "wind_speed": 18, "wind_dir": 270,
        "temp_c": -5, "payload_type": "oxytocin",
        "eta_min": 25, "nearest_lz": "LZ_03",
        "anomaly": "Battery drain rate doubled; headwind increased to 18 m/s",
    }

    decision = brain.in_flight_decision(telemetry, plan)

    print("\n[DECISION] KAIROS FLIGHT DECISION SUMMARY:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
