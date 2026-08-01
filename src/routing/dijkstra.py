"""
The Kairos Engine - Energy-Aware Weighted Dijkstra Pathfinding Engine

Production graph-based pathfinder using NetworkX. Builds a waypoint grid
along the Pokhara→Muktinath corridor with lateral alternatives, then
finds the minimum-energy path using Dijkstra's algorithm.
"""

import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("kairos.routing")


class KairosEnergyRouter:
    """
    Weighted Dijkstra Pathfinding Engine for BVLOS Drones in High-Altitude Terrain.
    Edge energy cost: E = k1(distance) + k2(altitude_gain) + k3(headwind) + k4(temp_drop)
    """

    # Energy cost coefficients
    K_DISTANCE  = 25.0    # Wh per km
    K_ALTITUDE  = 0.15    # Wh per meter climb
    K_WIND      = 0.50    # Wind drag coefficient
    K_TEMP      = 0.02    # Wh per degree below 15°C

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
        """Calculate energy cost for traversing an edge between two points."""
        dist_km = KairosEnergyRouter.haversine(p1, p2)

        base_cost = dist_km * KairosEnergyRouter.K_DISTANCE
        altitude_penalty = max(0, alt_gain) * KairosEnergyRouter.K_ALTITUDE
        wind_penalty = (wind_speed ** 1.8) * KairosEnergyRouter.K_WIND
        temp_penalty = max(0, 15 - temp_c) * KairosEnergyRouter.K_TEMP * dist_km

        return round(base_cost + altitude_penalty + wind_penalty + temp_penalty, 2)

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
                    "flight_time_min": round(total_energy / 35.0, 1),
                    "max_altitude": 4500,
                    "energy_margin_pct": round(max(0, (800 - total_energy) / 800 * 100), 1),
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
            "flight_time_min": round(total_energy / 35.0, 1),
            "max_altitude": 3850,
            "energy_margin_pct": round(max(0, (800 - total_energy) / 800 * 100), 1),
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
                    "flight_time_min": round(total_energy / 35.0, 1),
                })
            return results
        except Exception:
            return [cls._linear_route(origin, destination, wind_speed)]
