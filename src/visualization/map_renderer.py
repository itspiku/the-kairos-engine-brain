"""
The Kairos Engine - Interactive Folium GIS Map Renderer
"""

import os
from typing import List, Dict, Any


class KairosMapRenderer:
    """Generates high-contrast interactive OpenStreetMap HTML visualizers for flight corridors."""

    @staticmethod
    def render_flight_map(waypoints: List[List[float]], origin: List[float], destination: List[float], output_filename: str = "flight_plan_map.html") -> str:
        try:
            import folium
            center_lat = (origin[0] + destination[0]) / 2
            center_lon = (origin[1] + destination[1]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")

            # Route polyline
            folium.PolyLine(waypoints, color="#E63946", weight=4, opacity=0.85, tooltip="Kairos Energy-Aware BVLOS Flight Corridor").add_to(m)

            # Waypoint markers
            folium.Marker(origin, popup="<b>Origin</b>: Pokhara Airfield Base (28.2096, 83.9856)", icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(destination, popup="<b>Destination</b>: Muktinath Medical Clinic (28.8167, 83.8667)", icon=folium.Icon(color="red", icon="flag")).add_to(m)

            # Contingency LZs
            lzs = [
                {"name": "LZ_01 (High-Altitude Base)", "pos": [28.3500, 83.8800]},
                {"name": "LZ_02 (Jomsom Emergency Strip)", "pos": [28.7800, 83.7200]},
                {"name": "LZ_03 (Thorong Phedi Shelter)", "pos": [28.7900, 83.9400]}
            ]
            for lz in lzs:
                folium.Marker(lz["pos"], popup=f"<b>Contingency LZ</b>: {lz['name']}", icon=folium.Icon(color="orange", icon="info-sign")).add_to(m)

            m.save(output_filename)
            return output_filename
        except Exception as e:
            return ""
