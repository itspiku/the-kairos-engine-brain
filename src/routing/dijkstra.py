"""
The Kairos Engine - Energy-Aware Weighted Dijkstra Pathfinding Engine

Production graph-based pathfinder using NetworkX. Builds a waypoint grid
along the Pokhara→Muktinath corridor with lateral alternatives, then
finds the minimum-energy path using Dijkstra's algorithm.
"""

import math
import logging
from typing import List, Dict, Any, Optional

from src.config import (
    BATTERY_CAPACITY_WH, CRUISE_SPEED_MS, ENERGY_PER_KM_WH,
    ENERGY_PER_CLIMB_M_WH, ENERGY_PER_KM_PER_DEG, WIND_ADVERSE_FRACTION,
)

logger = logging.getLogger("kairos.routing")


class KairosEnergyRouter:
    """
    Weighted Dijkstra Pathfinding Engine for BVLOS Drones in High-Altitude Terrain.
    Edge energy cost: E = k1(distance) + k2(altitude_gain) + k3(headwind) + k4(temp_drop)

    Every term scales with the length of the edge it is charged against, so the
    total cost of a route is independent of how finely the corridor is segmented.
    """

    # Energy cost coefficients — see src/config.py for how each is derived.
    K_DISTANCE  = ENERGY_PER_KM_WH        # Wh per km of level cruise
    K_ALTITUDE  = ENERGY_PER_CLIMB_M_WH   # Wh per metre of climb
    K_WIND      = WIND_ADVERSE_FRACTION   # Share of wind treated as headwind
    K_TEMP      = ENERGY_PER_KM_PER_DEG   # Wh per km per °C below 15°C
    CRUISE_MS   = CRUISE_SPEED_MS         # Cruise airspeed the burn rate assumes
    CAPACITY_WH = BATTERY_CAPACITY_WH     # Usable pack energy

    @staticmethod
    def haversine(p1: List[float], p2: List[float]) -> float:
        """Great-circle distance in km between two [lat, lon] points."""
        R = 6371.0
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def calculate_edge_cost(p1: List[float], p2: List[float],
                            wind_speed: float = 0.0,
                            alt_gain: float = 0.0,
                            temp_c: float = 15.0) -> float:
        """
        Calculate energy cost in Wh for traversing an edge between two points.

        The wind term is charged per kilometre, not per edge. It previously used
        `wind_speed ** 1.8` with no distance factor, so identical wind cost the
        same energy over 1 km as over 55 km — which made the total energy of a
        route depend on how many segments the corridor was cut into.

        Headwind raises the burn rate because ground speed falls while airspeed
        power stays roughly constant, so the penalty is modelled as a fractional
        uplift on cruise burn proportional to wind speed over cruise speed.
        """
        cls = KairosEnergyRouter
        dist_km = cls.haversine(p1, p2)

        base_cost = dist_km * cls.K_DISTANCE
        altitude_penalty = max(0.0, alt_gain) * cls.K_ALTITUDE
        wind_ratio = max(0.0, wind_speed) / cls.CRUISE_MS
        wind_penalty = cls.K_WIND * wind_ratio * dist_km * cls.K_DISTANCE
        temp_penalty = max(0.0, 15 - temp_c) * cls.K_TEMP * dist_km

        return round(base_cost + altitude_penalty + wind_penalty + temp_penalty, 2)

    @classmethod
    def flight_time_min(cls, distance_km: float) -> float:
        """
        Convert route distance to flight time in minutes.

        Previously computed as `energy_wh / 35.0`, where 35 is documented
        elsewhere in this codebase as Wh **per km** — so the result was a
        distance in kilometres reported in a field named minutes.
        """
        return round(distance_km / cls.CRUISE_MS * 1000.0 / 60.0, 1)

    @classmethod
    def _build_corridor_graph(cls, origin: List[float], destination: List[float],
                              n_segments: int = 8, n_lateral: int = 3,
                              wind_speed: float = 12.0):
        """
        Build a waypoint graph along the flight corridor.
        Creates n_segments × n_lateral grid of nodes with energy-weighted edges.
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not available, falling back to linear routing")
            return None, None

        G = nx.DiGraph()
        nodes = {}  # (segment, lateral) -> [lat, lon]

        # Known elevation estimates along the corridor
        def estimate_elevation(lat, lon):
            # Simplified elevation model for the Pokhara-Muktinath corridor
            progress = (lat - origin[0]) / max(destination[0] - origin[0], 0.001)
            progress = max(0, min(1, progress))
            # Elevation profile: 800m → 3200m (midpoint peak) → 3800m (destination)
            if progress < 0.4:
                return 800 + progress / 0.4 * 2400
            elif progress < 0.7:
                return 3200 + (progress - 0.4) / 0.3 * 1300
            else:
                return 4500 - (progress - 0.7) / 0.3 * 700

        # Generate grid nodes
        lateral_offsets = [0.0]
        if n_lateral >= 3:
            lateral_offsets = [-0.02, 0.0, 0.02]  # ~2km lateral offsets
        elif n_lateral >= 2:
            lateral_offsets = [-0.01, 0.01]

        # Add origin
        G.add_node("origin", pos=origin)
        nodes["origin"] = origin

        # Add intermediate grid nodes
        for seg in range(1, n_segments):
            frac = seg / n_segments
            base_lat = origin[0] + frac * (destination[0] - origin[0])
            base_lon = origin[1] + frac * (destination[1] - origin[1])

            for lat_idx, offset in enumerate(lateral_offsets):
                node_id = f"s{seg}_l{lat_idx}"
                node_pos = [round(base_lat, 5), round(base_lon + offset, 5)]
                G.add_node(node_id, pos=node_pos)
                nodes[node_id] = node_pos

        # Add destination
        G.add_node("destination", pos=destination)
        nodes["destination"] = destination

        # Connect edges
        def add_edges_between_layers(from_nodes, to_nodes):
            for fn in from_nodes:
                fp = nodes[fn]
                fp_elev = estimate_elevation(fp[0], fp[1])
                for tn in to_nodes:
                    tp = nodes[tn]
                    tp_elev = estimate_elevation(tp[0], tp[1])
                    alt_gain = tp_elev - fp_elev
                    cost = cls.calculate_edge_cost(fp, tp, wind_speed=wind_speed, alt_gain=alt_gain)
                    G.add_edge(fn, tn, weight=cost, distance_km=cls.haversine(fp, tp))

        # Origin → first segment
        first_seg_nodes = [f"s1_l{i}" for i in range(len(lateral_offsets))]
        add_edges_between_layers(["origin"], first_seg_nodes)

        # Segment → next segment
        for seg in range(1, n_segments - 1):
            current = [f"s{seg}_l{i}" for i in range(len(lateral_offsets))]
            next_seg = [f"s{seg+1}_l{i}" for i in range(len(lateral_offsets))]
            add_edges_between_layers(current, next_seg)

        # Last segment → destination
        last_seg_nodes = [f"s{n_segments-1}_l{i}" for i in range(len(lateral_offsets))]
        add_edges_between_layers(last_seg_nodes, ["destination"])

        return G, nodes

    @classmethod
    def compute_route(cls, origin: List[float], destination: List[float],
                      drone_params: Dict[str, Any] = None,
                      wind: Dict = None) -> Dict[str, Any]:
        """
        Compute the energy-optimal route using graph-based Dijkstra.
        Falls back to linear interpolation if NetworkX unavailable.
        """
        wind_speed = wind.get("wind_speed", 12.0) if wind else 12.0

        G, nodes = cls._build_corridor_graph(origin, destination, wind_speed=wind_speed)

        if G is not None:
            try:
                import networkx as nx
                path = nx.dijkstra_path(G, "origin", "destination", weight="weight")
                waypoints = [nodes[n] for n in path]

                total_energy = sum(
                    G[path[i]][path[i+1]]["weight"]
                    for i in range(len(path) - 1)
                )
                total_distance = sum(
                    G[path[i]][path[i+1]].get("distance_km", 0)
                    for i in range(len(path) - 1)
                )

                route_segments = []
                for i in range(len(path) - 1):
                    route_segments.append({
                        "from": nodes[path[i]],
                        "to": nodes[path[i+1]],
                        "energy_wh": G[path[i]][path[i+1]]["weight"],
                        "distance_km": round(G[path[i]][path[i+1]].get("distance_km", 0), 2),
                    })

                return {
                    "waypoints": waypoints,
                    "estimated_energy_wh": round(total_energy, 1),
                    "flight_time_min": cls.flight_time_min(total_distance),
                    "max_altitude": 4500,
                    "energy_margin_pct": round(
                        (cls.CAPACITY_WH - total_energy) / cls.CAPACITY_WH * 100, 1),
                    "route_segments": route_segments,
                    "total_distance_km": round(total_distance, 1),
                    "algorithm": "dijkstra_networkx",
                }
            except Exception as exc:
                logger.warning(f"Dijkstra failed: {exc}, using linear fallback")

        # Fallback: linear interpolation with midpoints
        return cls._linear_route(origin, destination, wind_speed)

    @classmethod
    def _linear_route(cls, origin, destination, wind_speed=12.0):
        """Fallback linear route computation."""
        mid1 = [
            origin[0] + (destination[0] - origin[0]) * 0.33,
            origin[1] + (destination[1] - origin[1]) * 0.25,
        ]
        mid2 = [
            origin[0] + (destination[0] - origin[0]) * 0.66,
            origin[1] + (destination[1] - origin[1]) * 0.75,
        ]
        waypoints = [origin, mid1, mid2, destination]
        total_energy = sum(
            cls.calculate_edge_cost(waypoints[i], waypoints[i+1], wind_speed=wind_speed)
            for i in range(len(waypoints) - 1)
        )
        total_dist = sum(
            cls.haversine(waypoints[i], waypoints[i+1])
            for i in range(len(waypoints) - 1)
        )

        return {
            "waypoints": waypoints,
            "estimated_energy_wh": round(total_energy, 1),
            "flight_time_min": cls.flight_time_min(total_dist),
            "max_altitude": 3850,
            "energy_margin_pct": round(
                (cls.CAPACITY_WH - total_energy) / cls.CAPACITY_WH * 100, 1),
            "route_segments": [],
            "total_distance_km": round(total_dist, 1),
            "algorithm": "linear_fallback",
        }

    @classmethod
    def compute_k_shortest(cls, origin: List[float], destination: List[float],
                           k: int = 3, wind: Dict = None) -> List[Dict[str, Any]]:
        """Compute k shortest (lowest energy) route alternatives."""
        wind_speed = wind.get("wind_speed", 12.0) if wind else 12.0
        G, nodes = cls._build_corridor_graph(origin, destination, wind_speed=wind_speed)

        if G is None:
            return [cls._linear_route(origin, destination, wind_speed)]

        try:
            import networkx as nx
            paths = list(nx.shortest_simple_paths(G, "origin", "destination", weight="weight"))
            results = []
            for path in paths[:k]:
                waypoints = [nodes[n] for n in path]
                total_energy = sum(
                    G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1)
                )
                total_dist = sum(
                    G[path[i]][path[i+1]].get("distance_km", 0) for i in range(len(path)-1)
                )
                results.append({
                    "waypoints": waypoints,
                    "estimated_energy_wh": round(total_energy, 1),
                    "total_distance_km": round(total_dist, 1),
                    "flight_time_min": cls.flight_time_min(total_dist),
                })
            return results
        except Exception:
            return [cls._linear_route(origin, destination, wind_speed)]
