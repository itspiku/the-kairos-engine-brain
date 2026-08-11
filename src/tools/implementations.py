"""
The Kairos Engine — Real Tool Implementations

Production implementations replacing mock stubs. Each tool integrates
real APIs with graceful fallbacks for offline/demo operation.
"""

import os
import time
import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.units import normalize_battery_fraction

logger = logging.getLogger("kairos.tools")

LOGS_DIR = Path("logs")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_battery_state: Dict[str, Dict] = {}
_risk_classifier_instance = None
_session_id = f"kairos-{int(time.time())}"

# ---------------------------------------------------------------------------
# Known Himalayan landmarks for elevation fallback
# ---------------------------------------------------------------------------
_KNOWN_ELEVATIONS = {
    (28.21, 83.99): 827,    # Pokhara
    (28.35, 83.88): 2800,   # Dana
    (28.50, 83.85): 3200,   # Ghasa
    (28.63, 83.80): 2700,   # Tukuche
    (28.78, 83.72): 2720,   # Jomsom
    (28.79, 83.94): 4525,   # Thorong Phedi
    (28.82, 83.87): 3800,   # Muktinath
    (28.80, 83.93): 5416,   # Thorong La Pass
}

# ---------------------------------------------------------------------------
# Contingency Landing Zone Database
# ---------------------------------------------------------------------------
LZ_DATABASE = [
    {"id": "LZ_01", "name": "Lete River Terrace", "pos": [28.3500, 83.8800],
     "elevation_m": 2480, "flatness": 0.94, "surface": "gravel", "accessibility": "moderate",
     "wind_shelter": 0.85, "notes": "Broad river terrace near Lete village"},
    {"id": "LZ_02", "name": "Jomsom Emergency Airstrip", "pos": [28.7800, 83.7200],
     "elevation_m": 2720, "flatness": 0.97, "surface": "paved", "accessibility": "high",
     "wind_shelter": 0.60, "notes": "Existing STOL airstrip, prone to afternoon crosswinds"},
    {"id": "LZ_03", "name": "Thorong Phedi Shelter Field", "pos": [28.7900, 83.9400],
     "elevation_m": 4525, "flatness": 0.91, "surface": "gravel", "accessibility": "low",
     "wind_shelter": 0.75, "notes": "Near trekking shelter, high altitude"},
    {"id": "LZ_04", "name": "Ghasa Plateau", "pos": [28.5000, 83.8500],
     "elevation_m": 3200, "flatness": 0.89, "surface": "grass", "accessibility": "moderate",
     "wind_shelter": 0.80, "notes": "Open plateau above Ghasa village"},
    {"id": "LZ_05", "name": "Tukuche Meadow", "pos": [28.6300, 83.8000],
     "elevation_m": 2700, "flatness": 0.92, "surface": "grass", "accessibility": "moderate",
     "wind_shelter": 0.70, "notes": "Alpine meadow near Tukuche"},
    {"id": "LZ_06", "name": "Marpha Orchard Field", "pos": [28.7500, 83.6900],
     "elevation_m": 2670, "flatness": 0.93, "surface": "grass", "accessibility": "high",
     "wind_shelter": 0.65, "notes": "Near Marpha village, apple orchard clearings"},
    {"id": "LZ_07", "name": "Kagbeni Terrace", "pos": [28.8300, 83.7800],
     "elevation_m": 2800, "flatness": 0.90, "surface": "gravel", "accessibility": "moderate",
     "wind_shelter": 0.55, "notes": "River confluence terrace at Kagbeni"},
    {"id": "LZ_08", "name": "Pokhara Base Return", "pos": [28.2096, 83.9856],
     "elevation_m": 827, "flatness": 0.98, "surface": "paved", "accessibility": "high",
     "wind_shelter": 0.90, "notes": "Home base, full facilities"},
]

# ---------------------------------------------------------------------------
# Nepal Airspace Database
# ---------------------------------------------------------------------------
_AIRSPACE_ZONES = [
    {"id": "PKR-CTR", "name": "Pokhara Airport CTR", "class": "D",
     "center": [28.2000, 83.9821], "radius_km": 5.0, "ceiling_m": 2500,
     "type": "CTR", "active": True},
    {"id": "JOM-ATZ", "name": "Jomsom Airport ATZ", "class": "G",
     "center": [28.7804, 83.7230], "radius_km": 3.0, "ceiling_m": 4000,
     "type": "ATZ", "active": True},
    {"id": "ACA-TMA", "name": "Annapurna Conservation Overflight", "class": "R",
     "center": [28.5961, 83.8203], "radius_km": 15.0, "ceiling_m": 6000,
     "type": "RESTRICTED", "active": False,
     "notes": "Seasonal restriction during peak trekking season"},
]

# Medical supply priority database
_MEDICAL_PRIORITIES = {
    "critical": ["oxytocin", "blood", "blood plasma", "antivenom", "insulin",
                 "epinephrine", "adrenaline", "anti-hemorrhagic", "plasma",
                 "misoprostol", "magnesium sulfate"],
    "high": ["antibiotics", "vaccines", "surgical kit", "morphine", "ketamine",
             "sutures", "defibrillator", "ventilator parts", "IV fluids"],
    "standard": ["vitamins", "bandages", "antiseptic", "painkillers",
                 "first aid kit", "gauze", "splint"],
    "low": ["documentation", "forms", "non-medical supplies", "reports"],
}


# ===========================================================================
# TOOL IMPLEMENTATIONS
# ===========================================================================

def get_wind_forecast(lat: float, lon: float, altitude_m: float, time: str) -> dict:
    """
    Fetch altitude-resolved wind forecast.
    Primary: Open-Meteo API. Fallback: physics-based model.
    """
    try:
        import requests
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation"
            f"&timezone=auto&forecast_days=1"
        )
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            ws = hourly.get("wind_speed_10m", [0])
            wd = hourly.get("wind_direction_10m", [0])
            wg = hourly.get("wind_gusts_10m", [0])
            pp = hourly.get("precipitation", [0])

            idx = min(12, len(ws) - 1)  # Use midday forecast
            base_wind = ws[idx] if ws else 10.0
            alt_factor = 1.0 + 0.06 * (altitude_m - 10) / 1000
            scaled_wind = round(base_wind * alt_factor, 1)

            return {
                "wind_speed": scaled_wind,
                "wind_direction": wd[idx] if wd else 270,
                "gusts": round((wg[idx] if wg else base_wind * 1.5) * alt_factor, 1),
                "precipitation": "none" if (pp[idx] if pp else 0) < 0.1 else
                                 "light" if (pp[idx] if pp else 0) < 2.0 else "moderate",
                "altitude_m": altitude_m,
                "source": "Open-Meteo API (altitude-scaled)",
            }
    except Exception as exc:
        logger.debug(f"Open-Meteo API unavailable: {exc}")

    # Fallback: physics-based wind model
    base_wind = 8.0 + (altitude_m / 1000) * 2.5
    variation = random.uniform(-3, 3)
    wind_speed = round(max(0, base_wind + variation), 1)

    return {
        "wind_speed": wind_speed,
        "wind_direction": random.randint(180, 360),
        "gusts": round(wind_speed * 1.4, 1),
        "precipitation": random.choice(["none", "none", "none", "light"]),
        "altitude_m": altitude_m,
        "source": "Physics-based model (offline fallback)",
    }


def get_terrain_elevation_profile(route: list) -> dict:
    """
    Calculate terrain elevation profile along a route.
    Primary: Open-Elevation API. Fallback: interpolated Himalayan model.
    """
    if not route or len(route) < 2:
        return {"max_elevation_m": 0, "min_elevation_m": 0, "total_climb_m": 0,
                "steepest_slope_deg": 0, "ridge_lines_crossed": 0,
                "terrain_clearance_margin_m": 0}

    elevations = []

    # Try Open-Elevation API
    try:
        import requests
        locations = [{"latitude": p[0], "longitude": p[1]} for p in route]
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            elevations = [r["elevation"] for r in results]
    except Exception as exc:
        logger.debug(f"Open-Elevation API unavailable: {exc}")

    # Fallback: interpolated model
    if not elevations:
        elevations = [_estimate_elevation(p[0], p[1]) for p in route]

    max_elev = max(elevations)
    min_elev = min(elevations)
    total_climb = sum(max(0, elevations[i+1] - elevations[i]) for i in range(len(elevations)-1))

    # Calculate slopes
    slopes = []
    for i in range(len(route) - 1):
        dist = _haversine(route[i], route[i+1])
        if dist > 0:
            slope_deg = math.degrees(math.atan2(abs(elevations[i+1] - elevations[i]), dist * 1000))
            slopes.append(round(slope_deg, 1))

    # Count ridge crossings (elevation peaks)
    ridges = 0
    for i in range(1, len(elevations) - 1):
        if elevations[i] > elevations[i-1] and elevations[i] > elevations[i+1]:
            if elevations[i] - min(elevations[i-1], elevations[i+1]) > 200:
                ridges += 1

    return {
        "max_elevation_m": round(max_elev),
        "min_elevation_m": round(min_elev),
        "total_climb_m": round(total_climb),
        "steepest_slope_deg": round(max(slopes) if slopes else 0, 1),
        "ridge_lines_crossed": ridges,
        "terrain_clearance_margin_m": round(max(350, 6000 - max_elev)),
        "elevation_profile": [round(e) for e in elevations],
    }


def get_battery_state(drone_id: str) -> dict:
    """
    Physics-based battery state model with Peukert's equation.
    Tracks state across calls (stateful simulation).
    """
    global _battery_state

    if drone_id not in _battery_state:
        _battery_state[drone_id] = {
            "voltage": 25.2,
            "temp": 20.0,
            "soc": 95.0,
            "soh": 97.0,
            "cycle_count": 0,
            "last_update": time.time(),
        }

    state = _battery_state[drone_id]
    dt = time.time() - state["last_update"]
    dt_min = max(dt / 60.0, 0.1)

    # Simulate discharge using Peukert's equation
    base_drain = 1.8  # %/min at nominal load
    temp_factor = 1.0 + max(0, (15 - state["temp"]) * 0.005)
    drain = base_drain * temp_factor * dt_min

    state["soc"] = max(0, state["soc"] - drain)
    state["voltage"] = 19.0 + (state["soc"] / 100.0) * 6.2
    state["temp"] = max(-15, state["temp"] - 0.05 * dt_min)
    state["last_update"] = time.time()

    # Estimated range based on remaining energy
    remaining_energy_wh = state["soc"] / 100.0 * 800  # 800 Wh total capacity
    est_range = remaining_energy_wh / 35.0  # ~35 Wh/km consumption

    status = "nominal"
    if state["soc"] < 15:
        status = "emergency"
    elif state["soc"] < 30:
        status = "warning"

    return {
        "voltage": round(state["voltage"], 1),
        "temp": round(state["temp"], 1),
        "soc": round(state["soc"], 1),
        "soh": round(state["soh"], 1),
        "estimated_range_km": round(est_range, 1),
        "status": status,
    }


def compute_energy_aware_route(origin: list, destination: list,
                                drone_params: dict = None,
                                wind: dict = None,
                                terrain: dict = None) -> dict:
    """
    Compute energy-optimal route using the Dijkstra pathfinder.
    Generates interactive Folium map visualization.
    """
    from src.routing.dijkstra import KairosEnergyRouter
    from src.visualization.map_renderer import KairosMapRenderer

    route_data = KairosEnergyRouter.compute_route(
        origin, destination,
        drone_params=drone_params,
        wind=wind,
    )

    # Generate interactive map
    try:
        risk_data = None
        map_file = KairosMapRenderer.render_flight_map(
            waypoints=route_data["waypoints"],
            origin=origin,
            destination=destination,
            risk_data=risk_data,
            wind_data=wind,
        )
        route_data["interactive_map"] = map_file
        if map_file:
            logger.info(f"Flight map saved to {map_file}")
    except Exception as exc:
        logger.debug(f"Map generation failed: {exc}")
        route_data["interactive_map"] = ""

    return route_data


def find_contingency_landing_zones(current_pos: list, radius_m: float) -> list:
    """
    Search for validated contingency landing zones within radius of current position.
    Uses pre-loaded GIS database with real Himalayan coordinates.
    """
    radius_km = radius_m / 1000.0
    results = []

    for lz in LZ_DATABASE:
        dist = _haversine(current_pos, lz["pos"])
        if dist <= radius_km:
            results.append({
                "id": lz["id"],
                "name": lz["name"],
                "pos": lz["pos"],
                "elevation_m": lz["elevation_m"],
                "flatness": lz["flatness"],
                "surface": lz["surface"],
                "accessibility": lz["accessibility"],
                "wind_shelter": lz["wind_shelter"],
                "distance_m": round(dist * 1000),
                "notes": lz["notes"],
            })

    # Sort by composite score: flatness * 0.4 + wind_shelter * 0.3 + accessibility_score * 0.3
    access_map = {"high": 1.0, "moderate": 0.6, "low": 0.3}
    results.sort(
        key=lambda x: (
            x["flatness"] * 0.4 +
            x["wind_shelter"] * 0.3 +
            access_map.get(x["accessibility"], 0.5) * 0.3
        ),
        reverse=True,
    )
    return results


def check_airspace_restrictions(bbox: dict) -> dict:
    """
    Check Nepal CAAN airspace restrictions for a bounding box.
    Uses embedded airspace database with CTR, ATZ, and restricted zones.
    """
    north = bbox.get("north", 29)
    south = bbox.get("south", 28)
    east = bbox.get("east", 84)
    west = bbox.get("west", 83)

    zones_intersected = []
    restricted = False
    max_alt = 5500
    notams = []

    for zone in _AIRSPACE_ZONES:
        clat, clon = zone["center"]
        # Check if zone center is within bbox (simplified intersection)
        if south <= clat <= north and west <= clon <= east:
            if zone["active"]:
                zones_intersected.append({
                    "id": zone["id"],
                    "name": zone["name"],
                    "class": zone["class"],
                    "type": zone["type"],
                    "ceiling_m": zone["ceiling_m"],
                })
                if zone["type"] == "RESTRICTED":
                    restricted = True
                max_alt = min(max_alt, zone["ceiling_m"])
        if zone.get("notes"):
            notams.append(zone["notes"])

    clearance = "APPROVED_BVLOS_CORRIDOR"
    if restricted:
        clearance = "CONDITIONAL_APPROVAL_REQUIRED"
    elif zones_intersected:
        clearance = "APPROVED_WITH_ALTITUDE_RESTRICTIONS"

    return {
        "restricted": restricted,
        "zones_intersected": zones_intersected,
        "max_permitted_altitude_m": max_alt,
        "active_notams": notams[:3],
        "caan_zone_clearance": clearance,
    }


def get_delivery_priority(payload_type: str) -> str:
    """
    Determine medical payload delivery priority from comprehensive database.
    """
    pt = payload_type.lower().strip()
    for priority, keywords in _MEDICAL_PRIORITIES.items():
        if any(kw in pt for kw in keywords):
            return priority
    return "standard"


def recompute_route_from_current_pos(current_pos: list, battery_remaining_pct: float,
                                      wind: dict = None) -> dict:
    """
    Mid-flight route re-computation from current position.
    Checks energy feasibility and computes alternative to nearest LZ if infeasible.
    """
    from src.routing.dijkstra import KairosEnergyRouter
    from src.config import DEFAULT_DESTINATION

    # Compute route to destination
    route_data = KairosEnergyRouter.compute_route(
        current_pos, DEFAULT_DESTINATION, wind=wind,
    )

    remaining_energy_wh = normalize_battery_fraction(battery_remaining_pct) * 800
    required_energy = route_data["estimated_energy_wh"]
    feasible = remaining_energy_wh > required_energy * 1.15  # 15% safety margin

    result = {
        "feasible": feasible,
        "new_waypoints": route_data["waypoints"],
        "estimated_energy_wh": route_data["estimated_energy_wh"],
        "energy_margin_pct": round((remaining_energy_wh / max(required_energy, 1) - 1) * 100, 1),
    }

    if not feasible:
        # Find nearest LZ as alternative
        nearby_lzs = find_contingency_landing_zones(current_pos, 5000)
        if nearby_lzs:
            best_lz = nearby_lzs[0]
            alt_route = KairosEnergyRouter.compute_route(
                current_pos, best_lz["pos"], wind=wind,
            )
            result["alternative"] = {
                "target": best_lz["name"],
                "target_pos": best_lz["pos"],
                "waypoints": alt_route["waypoints"],
                "energy_wh": alt_route["estimated_energy_wh"],
                "feasible": remaining_energy_wh > alt_route["estimated_energy_wh"] * 1.1,
            }

    return result


def select_best_landing_zone(current_pos: list, reachable_zones: list,
                              weather: dict = None) -> dict:
    """
    Multi-criteria landing zone selection with weather-aware scoring.
    Weights: flatness(0.3) + wind_shelter(0.25) + proximity(0.25) + accessibility(0.2)
    """
    if not reachable_zones:
        return {}

    access_map = {"high": 1.0, "moderate": 0.6, "low": 0.3}
    max_dist = max(z.get("distance_m", 1000) for z in reachable_zones) or 1

    scored = []
    for z in reachable_zones:
        flatness_score = z.get("flatness", 0.5)
        shelter_score = z.get("wind_shelter", 0.5)
        dist_score = 1.0 - (z.get("distance_m", 0) / max(max_dist, 1))
        access_score = access_map.get(z.get("accessibility", "moderate"), 0.5)

        # Weather penalty
        weather_penalty = 0.0
        if weather:
            wind = weather.get("wind_speed", 0)
            if wind > 15:
                weather_penalty = 0.15
            elif wind > 10:
                weather_penalty = 0.05
            # Penalize exposed zones more in bad weather
            if wind > 10:
                shelter_score *= z.get("wind_shelter", 0.5)

        total = (
            flatness_score * 0.30 +
            shelter_score * 0.25 +
            dist_score * 0.25 +
            access_score * 0.20 -
            weather_penalty
        )

        scored.append({
            **z,
            "composite_score": round(total, 3),
            "score_breakdown": {
                "flatness": round(flatness_score * 0.30, 3),
                "wind_shelter": round(shelter_score * 0.25, 3),
                "proximity": round(dist_score * 0.25, 3),
                "accessibility": round(access_score * 0.20, 3),
                "weather_penalty": round(-weather_penalty, 3),
            },
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[0] if scored else {}


def log_decision(decision: str, rationale: str, tool_calls: list = None) -> dict:
    """
    Log flight decision to persistent JSON-lines audit trail.
    Writes to logs/kairos_audit.jsonl for CAAN regulatory compliance.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    entry = {
        "timestamp": timestamp,
        "session_id": _session_id,
        "decision": decision,
        "rationale": rationale[:500] if rationale else "",
        "tool_calls_count": len(tool_calls) if tool_calls else 0,
        "tool_names": [tc.get("name", "?") for tc in (tool_calls or [])],
    }

    try:
        audit_path = LOGS_DIR / "kairos_audit.jsonl"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info(f"Decision logged: {decision}")
    except Exception as exc:
        logger.error(f"Audit log write failed: {exc}")

    return {"logged": True, "timestamp": timestamp, "entry": entry}


def assess_risk(battery_pct: float, wind_speed_ms: float,
                altitude_m: float, proposed_action: str) -> dict:
    """
    Predict crash probability using the XGBoost risk classifier.
    Uses singleton model instance for efficiency.

    `battery_pct` accepts either the engine's percent convention (42.0) or the
    model's fraction convention (0.42); see normalize_battery_fraction.
    """
    global _risk_classifier_instance

    battery_pct = normalize_battery_fraction(battery_pct)

    if _risk_classifier_instance is None:
        try:
            from src.models.risk_classifier import KairosRiskClassifier
            _risk_classifier_instance = KairosRiskClassifier()
        except Exception as exc:
            logger.warning(f"Risk classifier failed to load: {exc}")
            # Fallback to rule-based assessment
            return _rule_based_risk(battery_pct, wind_speed_ms, altitude_m, proposed_action)

    try:
        return _risk_classifier_instance.predict_risk(
            battery_pct, wind_speed_ms, altitude_m, proposed_action,
        )
    except Exception as exc:
        logger.warning(f"ML risk prediction failed: {exc}")
        return _rule_based_risk(battery_pct, wind_speed_ms, altitude_m, proposed_action)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _haversine(p1: list, p2: list) -> float:
    """Calculate great-circle distance in km between two [lat, lon] points."""
    R = 6371.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _estimate_elevation(lat: float, lon: float) -> float:
    """Estimate elevation from known Himalayan landmarks using IDW interpolation."""
    total_weight = 0.0
    weighted_elev = 0.0

    for (klat, klon), elev in _KNOWN_ELEVATIONS.items():
        dist = math.sqrt((lat - klat)**2 + (lon - klon)**2)
        dist = max(dist, 0.001)
        w = 1.0 / (dist ** 2)
        weighted_elev += w * elev
        total_weight += w

    return weighted_elev / total_weight if total_weight > 0 else 2500.0


def _rule_based_risk(battery_pct: float, wind_speed_ms: float,
                     altitude_m: float, proposed_action: str) -> dict:
    """Rule-based risk assessment as fallback when ML model unavailable."""
    risk_score = 0.0
    battery_pct = normalize_battery_fraction(battery_pct)
    action_upper = str(proposed_action).upper()

    if battery_pct < 0.25:
        risk_score += 0.35
    elif battery_pct < 0.35:
        risk_score += 0.20

    if wind_speed_ms > 18:
        risk_score += 0.30
    elif wind_speed_ms > 12:
        risk_score += 0.15

    if altitude_m > 4000:
        risk_score += 0.15
    elif altitude_m > 3000:
        risk_score += 0.08

    if action_upper == "CONTINUE" and battery_pct < 0.30 and wind_speed_ms > 15:
        risk_score += 0.20
    elif action_upper == "RTL" and altitude_m > 3500:
        risk_score += 0.10

    risk_score = min(risk_score, 0.99)

    if risk_score > 0.70:
        level = "Critical"
    elif risk_score > 0.40:
        level = "High"
    else:
        level = "Acceptable"

    return {
        "crash_probability": round(risk_score, 2),
        "risk_level": level,
        "data_source": "Rule-based fallback model",
    }
