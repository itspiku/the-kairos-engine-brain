import json
from src.brain import GemmaBrain

def main():
    # Initialize the Khumbu Engine
    brain = GemmaBrain()
    
    # ── TEST 1: Pre-flight Planning ──
    mission = (
        "Send oxytocin to the clinic in Muktinath (28.8167, 83.8667) "
        "before the storm hits in 90 minutes. Origin is Pokhara (28.2096, 83.9856)."
    )
    plan = brain.pre_flight_plan(mission)
    
    print("\n📋 FLIGHT PLAN SUMMARY:")
    print(json.dumps(plan, indent=2))
    
    # ── TEST 2: In-flight Anomaly ──
    telemetry = {
        "lat": 28.3500, "lon": 83.8800, "altitude_m": 3200,
        "battery_pct": 42, "drain_rate": 4.2,
        "wind_speed": 18, "wind_dir": 270,
        "temp_c": -5, "payload_type": "oxytocin",
        "eta_min": 25, "nearest_lz": "LZ_03",
        "anomaly": "Battery drain rate doubled; headwind increased to 18 m/s"
    }
    
    decision = brain.in_flight_decision(telemetry, plan)
    
    print("\n🎯 FLIGHT DECISION SUMMARY:")
    print(json.dumps(decision, indent=2))

if __name__ == "__main__":
    main()
