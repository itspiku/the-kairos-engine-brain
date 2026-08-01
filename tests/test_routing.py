"""
The Kairos Engine — Routing & Spatial Test Suite
"""

import os
import sys
import math
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routing.dijkstra import KairosEnergyRouter
from src.routing.spatial import haversine_distance, interpolate_route_points, get_elevation_profile
from src.config import DEFAULT_ORIGIN, DEFAULT_DESTINATION


class TestHaversineDistance(unittest.TestCase):

    def test_zero_distance_same_point(self):
        dist = haversine_distance([28.2096, 83.9856], [28.2096, 83.9856])
        self.assertAlmostEqual(dist, 0.0, places=3)

    def test_pokhara_to_muktinath_approx_68km(self):
        dist = haversine_distance(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        # Straight-line ~68km
        self.assertGreater(dist, 50)
        self.assertLess(dist, 90)

    def test_known_distance_positive(self):
        dist = haversine_distance([28.0, 83.0], [29.0, 84.0])
        self.assertGreater(dist, 0)

    def test_symmetry(self):
        d1 = haversine_distance(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        d2 = haversine_distance(DEFAULT_DESTINATION, DEFAULT_ORIGIN)
        self.assertAlmostEqual(d1, d2, places=5)


class TestRouteInterpolation(unittest.TestCase):

    def test_returns_correct_number_of_points(self):
        pts = interpolate_route_points(DEFAULT_ORIGIN, DEFAULT_DESTINATION, n_points=10)
        self.assertEqual(len(pts), 11)  # n_points + 1

    def test_first_point_is_origin(self):
        pts = interpolate_route_points(DEFAULT_ORIGIN, DEFAULT_DESTINATION, n_points=5)
        self.assertAlmostEqual(pts[0][0], DEFAULT_ORIGIN[0], places=4)
        self.assertAlmostEqual(pts[0][1], DEFAULT_ORIGIN[1], places=4)

    def test_last_point_is_destination(self):
        pts = interpolate_route_points(DEFAULT_ORIGIN, DEFAULT_DESTINATION, n_points=5)
        self.assertAlmostEqual(pts[-1][0], DEFAULT_DESTINATION[0], places=4)
        self.assertAlmostEqual(pts[-1][1], DEFAULT_DESTINATION[1], places=4)


class TestEnergyRouter(unittest.TestCase):

    def test_edge_cost_is_positive(self):
        cost = KairosEnergyRouter.calculate_edge_cost(
            DEFAULT_ORIGIN, DEFAULT_DESTINATION, wind_speed=10.0, alt_gain=1000
        )
        self.assertGreater(cost, 0)

    def test_edge_cost_increases_with_wind(self):
        cost_low = KairosEnergyRouter.calculate_edge_cost(
            DEFAULT_ORIGIN, DEFAULT_DESTINATION, wind_speed=5.0, alt_gain=0
        )
        cost_high = KairosEnergyRouter.calculate_edge_cost(
            DEFAULT_ORIGIN, DEFAULT_DESTINATION, wind_speed=20.0, alt_gain=0
        )
        self.assertGreater(cost_high, cost_low)

    def test_edge_cost_increases_with_altitude_gain(self):
        cost_flat = KairosEnergyRouter.calculate_edge_cost(
            DEFAULT_ORIGIN, DEFAULT_DESTINATION, wind_speed=10.0, alt_gain=0
        )
        cost_climb = KairosEnergyRouter.calculate_edge_cost(
            DEFAULT_ORIGIN, DEFAULT_DESTINATION, wind_speed=10.0, alt_gain=2000
        )
        self.assertGreater(cost_climb, cost_flat)

    def test_compute_route_returns_dict(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertIsInstance(result, dict)

    def test_route_has_required_keys(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        for key in ["waypoints", "estimated_energy_wh", "flight_time_min", "max_altitude"]:
            self.assertIn(key, result)

    def test_route_starts_at_origin(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        wp = result["waypoints"]
        self.assertGreaterEqual(len(wp), 2)
        self.assertAlmostEqual(wp[0][0], DEFAULT_ORIGIN[0], places=3)
        self.assertAlmostEqual(wp[0][1], DEFAULT_ORIGIN[1], places=3)

    def test_route_ends_at_destination(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        wp = result["waypoints"]
        self.assertAlmostEqual(wp[-1][0], DEFAULT_DESTINATION[0], places=3)
        self.assertAlmostEqual(wp[-1][1], DEFAULT_DESTINATION[1], places=3)

    def test_energy_is_positive(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreater(result["estimated_energy_wh"], 0)

    def test_flight_time_is_positive(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreater(result["flight_time_min"], 0)

    def test_minimum_four_waypoints(self):
        result = KairosEnergyRouter.compute_route(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreaterEqual(len(result["waypoints"]), 4)

    def test_haversine_static_method(self):
        dist = KairosEnergyRouter.haversine(DEFAULT_ORIGIN, DEFAULT_DESTINATION)
        self.assertGreater(dist, 50)
        self.assertLess(dist, 90)


class TestElevationProfile(unittest.TestCase):

    def test_profile_returns_dict(self):
        route = [DEFAULT_ORIGIN, DEFAULT_DESTINATION]
        result = get_elevation_profile(route)
        self.assertIsInstance(result, dict)

    def test_profile_has_required_keys(self):
        route = [DEFAULT_ORIGIN, [28.5, 83.9], DEFAULT_DESTINATION]
        result = get_elevation_profile(route)
        for key in ["max_elevation_m", "min_elevation_m", "total_climb_m",
                    "steepest_slope_deg", "ridge_lines_crossed"]:
            self.assertIn(key, result)

    def test_max_elevation_greater_than_min(self):
        route = [DEFAULT_ORIGIN, [28.5, 83.9], DEFAULT_DESTINATION]
        result = get_elevation_profile(route)
        self.assertGreaterEqual(result["max_elevation_m"], result["min_elevation_m"])

    def test_empty_route_graceful(self):
        result = get_elevation_profile([])
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
