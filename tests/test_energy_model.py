"""
The Kairos Engine — Energy Model Property Tests

Property tests over calculate_edge_cost and compute_route. Each of these would
have failed against the previous implementation:

- The wind term was `wind_speed ** 1.8` with no distance factor, charged once per
  edge. Identical wind cost the same energy over 1 km as over 55 km, so total
  route energy scaled with the number of segments the corridor was cut into.
- `flight_time_min` was `energy_wh / 35.0`, where 35 is Wh per *kilometre* — it
  reported a distance in a field named minutes.
- 25 Wh/km against an 800 Wh pack implied a 32 km range on a 68.5 km corridor, so
  every mission computed as infeasible even at full charge.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routing.dijkstra import KairosEnergyRouter as Router
from src.config import (
    DEFAULT_ORIGIN, DEFAULT_DESTINATION, BATTERY_CAPACITY_WH,
    CRUISE_SPEED_MS, ENERGY_PER_KM_WH,
)

NORTH = [28.00, 84.00]
NEAR = [28.01, 84.00]     # ~1.1 km
FAR = [28.50, 84.00]      # ~55.6 km


class TestEdgeCostMonotonicity(unittest.TestCase):
    """Energy must never decrease when a cost driver increases."""

    def test_increases_with_distance(self):
        near = Router.calculate_edge_cost(NORTH, NEAR, wind_speed=10)
        far = Router.calculate_edge_cost(NORTH, FAR, wind_speed=10)
        self.assertGreater(far, near)

    def test_increases_with_wind(self):
        costs = [Router.calculate_edge_cost(NORTH, FAR, wind_speed=w)
                 for w in [0, 5, 10, 15, 20, 25]]
        for lower, higher in zip(costs, costs[1:]):
            self.assertLess(lower, higher)

    def test_increases_with_climb(self):
        costs = [Router.calculate_edge_cost(NORTH, FAR, alt_gain=g)
                 for g in [0, 500, 1000, 2000, 3000]]
        for lower, higher in zip(costs, costs[1:]):
            self.assertLess(lower, higher)

    def test_increases_as_temperature_drops(self):
        costs = [Router.calculate_edge_cost(NORTH, FAR, temp_c=t)
                 for t in [15, 5, -5, -15, -25]]
        for lower, higher in zip(costs, costs[1:]):
            self.assertLess(lower, higher)

    def test_descent_is_not_cheaper_than_level(self):
        """Negative altitude gain must not create an energy credit."""
        level = Router.calculate_edge_cost(NORTH, FAR, alt_gain=0)
        descent = Router.calculate_edge_cost(NORTH, FAR, alt_gain=-2000)
        self.assertAlmostEqual(level, descent, places=2)

    def test_zero_distance_costs_nothing_without_climb(self):
        self.assertAlmostEqual(
            Router.calculate_edge_cost(NORTH, NORTH, wind_speed=20), 0.0, places=2)


class TestWindScalesWithDistance(unittest.TestCase):
    """The specific defect: the wind term ignored edge length."""

    def test_wind_penalty_is_proportional_to_distance(self):
        short_delta = (Router.calculate_edge_cost(NORTH, NEAR, wind_speed=12)
                       - Router.calculate_edge_cost(NORTH, NEAR, wind_speed=0))
        long_delta = (Router.calculate_edge_cost(NORTH, FAR, wind_speed=12)
                      - Router.calculate_edge_cost(NORTH, FAR, wind_speed=0))

        dist_ratio = Router.haversine(NORTH, FAR) / Router.haversine(NORTH, NEAR)
        wind_ratio = long_delta / short_delta

        self.assertAlmostEqual(
            wind_ratio, dist_ratio, delta=dist_ratio * 0.02,
            msg="wind energy must scale with edge length, not per-edge",
        )

    def test_route_energy_is_stable_under_resegmentation(self):
        """
        Total energy must not depend on how finely the corridor is sliced.
        Previously the per-edge wind charge made a 32-segment route cost ~8x the
        wind energy of a 4-segment one over the same ground.
        """
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("networkx not installed")

        energies = []
        for n in [4, 8, 16, 32]:
            G, nodes = Router._build_corridor_graph(
                DEFAULT_ORIGIN, DEFAULT_DESTINATION, n_segments=n, wind_speed=12.0)
            path = nx.dijkstra_path(G, "origin", "destination", weight="weight")
            energies.append(sum(G[path[i]][path[i + 1]]["weight"]
                                for i in range(len(path) - 1)))

        spread = (max(energies) - min(energies)) / min(energies)
        self.assertLess(spread, 0.05,
                        f"energy varies {spread:.1%} with segmentation: {energies}")


class TestFlightTimeUnits(unittest.TestCase):

    def test_flight_time_matches_cruise_speed(self):
        distance_km = 72.0
        expected_min = distance_km / (CRUISE_SPEED_MS * 3.6) * 60
        self.assertAlmostEqual(Router.flight_time_min(distance_km), expected_min, delta=0.2)

    def test_route_implied_speed_is_the_cruise_speed(self):
        route = Router.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        implied_kmh = route["total_distance_km"] / (route["flight_time_min"] / 60.0)
        self.assertAlmostEqual(implied_kmh, CRUISE_SPEED_MS * 3.6, delta=1.0)

    def test_flight_time_is_not_the_old_energy_over_35(self):
        route = Router.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertNotAlmostEqual(
            route["flight_time_min"], route["estimated_energy_wh"] / 35.0, delta=1.0)

    def test_flight_time_scales_with_distance(self):
        self.assertAlmostEqual(Router.flight_time_min(100), 2 * Router.flight_time_min(50), delta=0.2)


class TestCorridorIsFlyable(unittest.TestCase):
    """The default mission must be possible on a full charge."""

    def test_default_corridor_fits_in_the_battery(self):
        route = Router.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertLess(
            route["estimated_energy_wh"], BATTERY_CAPACITY_WH,
            "the flagship mission must not require more than a full pack",
        )

    def test_default_corridor_leaves_a_usable_reserve(self):
        route = Router.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreater(route["energy_margin_pct"], 15.0)

    def test_energy_margin_is_not_permanently_clamped_to_zero(self):
        route = Router.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertNotEqual(route["energy_margin_pct"], 0.0)

    def test_range_is_consistent_with_burn_rate(self):
        max_range_km = BATTERY_CAPACITY_WH / ENERGY_PER_KM_WH
        corridor_km = Router.haversine(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreater(max_range_km, corridor_km,
                           "still-air range must exceed the corridor it is built for")


class TestFeasibilityDiscriminates(unittest.TestCase):
    """recompute_route_from_current_pos previously returned False at every level."""

    def test_full_battery_is_feasible(self):
        from src.tools.implementations import recompute_route_from_current_pos
        self.assertTrue(recompute_route_from_current_pos([28.35, 83.88], 100.0)["feasible"])

    def test_low_battery_is_infeasible(self):
        from src.tools.implementations import recompute_route_from_current_pos
        self.assertFalse(recompute_route_from_current_pos([28.35, 83.88], 20.0)["feasible"])

    def test_infeasible_route_offers_an_alternative(self):
        from src.tools.implementations import recompute_route_from_current_pos
        result = recompute_route_from_current_pos([28.35, 83.88], 20.0)
        self.assertIn("alternative", result)
        self.assertIn("target", result["alternative"])

    def test_margin_falls_as_battery_falls(self):
        from src.tools.implementations import recompute_route_from_current_pos
        margins = [recompute_route_from_current_pos([28.35, 83.88], pct)["energy_margin_pct"]
                   for pct in [100, 80, 60, 40, 20]]
        for higher, lower in zip(margins, margins[1:]):
            self.assertGreater(higher, lower)


if __name__ == "__main__":
    unittest.main()
