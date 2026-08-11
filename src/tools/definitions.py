"""
Definitions (JSON schemas) for all tools available to the Khumbu Engine cognitive autopilot.
"""

from typing import List, Dict

def get_tool_definitions() -> List[Dict]:
    """All tools available to the cognitive autopilot."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_wind_forecast",
                "description": "Fetches real-time, altitude-resolved wind speed/direction and precipitation from Open-Meteo API for a given location and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number", "description": "Latitude"},
                        "lon": {"type": "number", "description": "Longitude"},
                        "altitude_m": {"type": "number", "description": "Altitude in meters"},
                        "time": {"type": "string", "description": "ISO 8601 timestamp"}
                    },
                    "required": ["lat", "lon", "altitude_m", "time"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_terrain_elevation_profile",
                "description": "Queries SRTM 30m DEM data to find altitude gains, terrain slopes, and ridge lines along a proposed route.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "List of [lat, lon] waypoints"
                        }
                    },
                    "required": ["route"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_battery_state",
                "description": "Reads simulated telemetry: voltage, temperature, state-of-charge (SoC), and state-of-health (SoH).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drone_id": {"type": "string", "description": "Drone identifier"}
                    },
                    "required": ["drone_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_energy_aware_route",
                "description": "Custom weighted-Dijkstra engine. Edge weights are calculated energy consumption: E = k1(dist) + k2(alt_gain) + k3(headwind) + k4(temp_drop). Returns safest, most energy-efficient route.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "array", "items": {"type": "number"}, "description": "[lat, lon]"},
                        "destination": {"type": "array", "items": {"type": "number"}, "description": "[lat, lon]"},
                        "drone_params": {"type": "object", "description": "Mass, battery capacity, motor efficiency"},
                        "wind": {"type": "object", "description": "Wind field data"},
                        "terrain": {"type": "object", "description": "Terrain elevation data"}
                    },
                    "required": ["origin", "destination", "drone_params"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_contingency_landing_zones",
                "description": "Queries pre-mapped safe zones: flat terrain, away from villages, within glide range of current position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_pos": {"type": "array", "items": {"type": "number"}, "description": "[lat, lon]"},
                        "radius_m": {"type": "number", "description": "Search radius in meters"}
                    },
                    "required": ["current_pos", "radius_m"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_airspace_restrictions",
                "description": "Checks CAAN (Civil Aviation Authority of Nepal) no-fly zones, NOTAMs, and altitude ceilings for a bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "north": {"type": "number"}, "south": {"type": "number"},
                                "east": {"type": "number"}, "west": {"type": "number"}
                            },
                            "required": ["north", "south", "east", "west"]
                        }
                    },
                    "required": ["bbox"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_delivery_priority",
                "description": "Returns payload criticality level: critical (oxytocin, blood plasma), standard (routine supplies), or low.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload_type": {"type": "string", "description": "Type of medical payload"}
                    },
                    "required": ["payload_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recompute_route_from_current_pos",
                "description": "Mid-flight re-planning from current position given remaining battery and updated wind conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_pos": {"type": "array", "items": {"type": "number"}},
                        "battery_remaining_pct": {"type": "number"},
                        "wind": {"type": "object"}
                    },
                    "required": ["current_pos", "battery_remaining_pct"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_best_landing_zone",
                "description": "Ranks reachable contingency landing zones by safety, weather shelter, and accessibility for ground crew.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_pos": {"type": "array", "items": {"type": "number"}},
                        "reachable_zones": {"type": "array", "items": {"type": "object"}},
                        "weather": {"type": "object"}
                    },
                    "required": ["current_pos", "reachable_zones"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "log_decision",
                "description": "Logs decision, rationale, and tool calls to audit trail for CAAN regulators.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string"},
                        "rationale": {"type": "string"},
                        "tool_calls": {"type": "array"}
                    },
                    "required": ["decision", "rationale"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "assess_risk",
                "description": "Predicts drone crash probability and risk level based on battery percentage, headwind speed, altitude, and proposed flight action using an XGBoost ML model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "battery_pct": {"type": "number", "description": "Battery remaining as a percentage, 0 to 100 (e.g. 42 for 42%)"},
                        "wind_speed_ms": {"type": "number", "description": "Headwind speed in meters/second (0 to 25)"},
                        "altitude_m": {"type": "integer", "description": "Current altitude in meters (1000 to 5000)"},
                        "proposed_action": {"type": "string", "description": "Proposed action: 'CONTINUE', 'RTL', or 'DIVERT'"}
                    },
                    "required": ["battery_pct", "wind_speed_ms", "altitude_m", "proposed_action"]
                }
            }
        }
    ]
