"""
The Kairos Engine — CLI Entry Point

Downloads Gemma 4 E2B Q4_K_M GGUF if not present, then runs the demo mission.
Supports CLI arguments for custom missions, origins, destinations, and verbosity.
"""

import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import argparse
from pathlib import Path
from src.config import (
    DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILENAME, MODEL_DIR,
    DEFAULT_ORIGIN, DEFAULT_DESTINATION,
)
from src.core.telemetry import TelemetryReport


def ensure_model(model_dir: Path = MODEL_DIR) -> str:
    """Download the GGUF model from Hugging Face if not present."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / DEFAULT_MODEL_FILENAME

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
        local_dir=str(model_dir),
    )
    print(f"\n[OK] Download complete: {local}\n")
    return local


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="The Kairos Engine — AI Cognitive Autopilot for BVLOS Medical Drones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python main.py --mission 'Deliver insulin to Jomsom' --verbose",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to the GGUF model file (default: auto-download)",
    )
    parser.add_argument(
        "--mission", type=str, default=None,
        help="Custom mission request text (default: oxytocin to Muktinath)",
    )
    parser.add_argument(
        "--origin", type=str, default=None,
        help="Origin coordinates as 'lat,lon' (default: Pokhara Base)",
    )
    parser.add_argument(
        "--destination", type=str, default=None,
        help="Destination coordinates as 'lat,lon' (default: Muktinath Clinic)",
    )
    parser.add_argument(
        "--no-inflight", action="store_true",
        help="Skip the in-flight anomaly phase (run pre-flight only)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse coordinates
    origin = DEFAULT_ORIGIN
    destination = DEFAULT_DESTINATION
    if args.origin:
        parts = args.origin.split(",")
        origin = [float(parts[0]), float(parts[1])]
    if args.destination:
        parts = args.destination.split(",")
        destination = [float(parts[0]), float(parts[1])]

    # Ensure model
    model_path = args.model_path
    if model_path is None:
        model_path = ensure_model()

    try:
        from src.brain import KairosBrain
        brain = KairosBrain(model_path=model_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize Kairos Engine: {exc}")
        return 1

    # ── MISSION A: Pre-flight Planning ──
    mission = args.mission or (
        f"Send oxytocin to the clinic in Muktinath ({destination[0]}, {destination[1]}) "
        f"before the storm hits in 90 minutes. Origin is Pokhara Base ({origin[0]}, {origin[1]})."
    )

    try:
        plan = brain.pre_flight_plan(mission, drone_id="KAIROS-01")
        print("\n[PLAN] KAIROS FLIGHT PLAN SUMMARY:")
        print(json.dumps(plan, indent=2, default=str))
    except Exception as exc:
        print(f"\n[ERROR] Pre-flight planning failed: {exc}")
        return 1

    # ── MISSION B: In-flight Anomaly Re-planning ──
    if not args.no_inflight:
        telemetry = TelemetryReport(
            lat=28.3500, lon=83.8800, altitude_m=3200,
            battery_pct=42, drain_rate=4.2,
            wind_speed=18, wind_dir=270,
            temp_c=-5, payload_type="oxytocin",
            eta_min=25, nearest_lz="LZ_03",
            anomaly="Battery drain rate doubled; headwind increased to 18 m/s",
        )

        try:
            decision = brain.in_flight_decision(telemetry, plan)
            print("\n[DECISION] KAIROS FLIGHT DECISION SUMMARY:")
            print(json.dumps(decision, indent=2, default=str))
        except Exception as exc:
            print(f"\n[ERROR] In-flight decision failed: {exc}")

        # Check SLA
        if decision and not decision.get("meets_sla", True):
            print("\n[WARNING] SLA VIOLATION: Decision exceeded 2.0s latency requirement.")

    # ── Post-flight report ──
    try:
        brain.post_flight_report()
    except Exception:
        pass

    # Graceful shutdown
    brain.shutdown()
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
