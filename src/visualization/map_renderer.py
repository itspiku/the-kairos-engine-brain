"""
The Kairos Engine - Interactive Folium GIS Map Renderer

Advanced visualization with CartoDB dark tiles, risk-graded route coloring,
wind barbs, decision markers, flight metrics panel, and layer controls.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("kairos.map")


class KairosMapRenderer:
    """Generates high-contrast interactive flight corridor visualizations."""

    # Contingency Landing Zone database for map rendering
    DEFAULT_LZS = [
        {"name": "LZ_01 Lete River Terrace", "pos": [28.3500, 83.8800], "elevation": 2480},
        {"name": "LZ_02 Jomsom Emergency Airstrip", "pos": [28.7800, 83.7200], "elevation": 2720},
        {"name": "LZ_03 Thorong Phedi Shelter", "pos": [28.7900, 83.9400], "elevation": 4525},
        {"name": "LZ_04 Ghasa Plateau", "pos": [28.5000, 83.8500], "elevation": 3200},
        {"name": "LZ_05 Tukuche Meadow", "pos": [28.6300, 83.8000], "elevation": 2700},
        {"name": "LZ_06 Marpha Orchard", "pos": [28.7500, 83.6900], "elevation": 2670},
        {"name": "LZ_07 Kagbeni Terrace", "pos": [28.8300, 83.7800], "elevation": 2800},
    ]

    @staticmethod
    def render_flight_map(waypoints: List[List[float]],
                          origin: List[float],
                          destination: List[float],
                          risk_data: Optional[List[Dict]] = None,
                          wind_data: Optional[Dict] = None,
                          decision_markers: Optional[List[Dict]] = None,
                          output_filename: str = "flight_plan_map.html") -> str:
        """
        Render a comprehensive interactive flight corridor map.

        Args:
            waypoints: List of [lat, lon] route waypoints
            origin: [lat, lon] of origin
            destination: [lat, lon] of destination
            risk_data: Optional list of {segment, risk_score} for risk coloring
            wind_data: Optional {wind_speed, wind_direction} for wind overlay
            decision_markers: Optional list of {pos, decision, rationale}
            output_filename: Output HTML file path

        Returns:
            Filename of generated map, or empty string on failure.
        """
        try:
            import folium
            from folium import plugins
        except ImportError:
            logger.warning("Folium not installed, skipping map generation")
            return ""

        try:
            # Center map
            center_lat = (origin[0] + destination[0]) / 2
            center_lon = (origin[1] + destination[1]) / 2

            # Dark tiles for dramatic, professional look
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=9,
                tiles="CartoDB dark_matter",
                attr="CartoDB",
            )

            # Add tile layer options
            folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Satellite",
            ).add_to(m)

            # ── Route polyline (risk-graded if data available) ──
            if risk_data and len(risk_data) >= len(waypoints) - 1:
                for i in range(len(waypoints) - 1):
                    score = risk_data[i].get("risk_score", 0.3) if i < len(risk_data) else 0.3
                    color = _risk_color(score)
                    folium.PolyLine(
                        [waypoints[i], waypoints[i + 1]],
                        color=color, weight=5, opacity=0.9,
                        tooltip=f"Segment {i+1}: Risk {score:.0%}",
                    ).add_to(m)
            else:
                folium.PolyLine(
                    waypoints, color="#E63946", weight=4, opacity=0.85,
                    tooltip="Kairos Energy-Aware BVLOS Flight Corridor",
                ).add_to(m)

            # ── Origin marker ──
            folium.Marker(
                origin,
                popup=folium.Popup(
                    f"<div style='font-family:monospace;'>"
                    f"<b>ORIGIN</b><br>"
                    f"Pokhara Airfield Base<br>"
                    f"({origin[0]:.4f}, {origin[1]:.4f})<br>"
                    f"Elevation: 827m AMSL</div>",
                    max_width=250,
                ),
                icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
                tooltip="Origin: Pokhara Base",
            ).add_to(m)

            # ── Destination marker ──
            folium.Marker(
                destination,
                popup=folium.Popup(
                    f"<div style='font-family:monospace;'>"
                    f"<b>DESTINATION</b><br>"
                    f"Muktinath Medical Clinic<br>"
                    f"({destination[0]:.4f}, {destination[1]:.4f})<br>"
                    f"Elevation: 3,800m AMSL</div>",
                    max_width=250,
                ),
                icon=folium.Icon(color="red", icon="flag", prefix="glyphicon"),
                tooltip="Destination: Muktinath Clinic",
            ).add_to(m)

            # ── Contingency Landing Zone markers ──
            lz_group = folium.FeatureGroup(name="Contingency LZs", show=True)
            for lz in KairosMapRenderer.DEFAULT_LZS:
                folium.Marker(
                    lz["pos"],
                    popup=folium.Popup(
                        f"<div style='font-family:monospace;'>"
                        f"<b>CONTINGENCY LZ</b><br>"
                        f"{lz['name']}<br>"
                        f"Elevation: {lz['elevation']}m</div>",
                        max_width=250,
                    ),
                    icon=folium.Icon(color="orange", icon="info-sign", prefix="glyphicon"),
                    tooltip=f"LZ: {lz['name']}",
                ).add_to(lz_group)
            lz_group.add_to(m)

            # ── Wind barb overlay ──
            if wind_data:
                wind_group = folium.FeatureGroup(name="Wind Data", show=True)
                ws = wind_data.get("wind_speed", 10)
                wd = wind_data.get("wind_direction", 270)
                # Place wind indicator at midpoint
                folium.Marker(
                    [center_lat, center_lon],
                    popup=f"Wind: {ws} m/s from {wd}deg",
                    icon=folium.DivIcon(html=(
                        f"<div style='color:#00BFFF;font-size:11px;font-weight:bold;"
                        f"text-shadow:1px 1px 2px black;'>"
                        f"&#x2794; {ws}m/s</div>"
                    )),
                    tooltip=f"Wind: {ws}m/s @ {wd}deg",
                ).add_to(wind_group)
                wind_group.add_to(m)

            # ── Decision markers ──
            if decision_markers:
                dec_group = folium.FeatureGroup(name="Decisions", show=True)
                dec_icons = {
                    "CONTINUE": ("green", "ok"),
                    "DIVERT": ("orange", "warning-sign"),
                    "LAND": ("red", "stop"),
                    "ABORT": ("darkred", "remove"),
                    "RTL": ("purple", "home"),
                }
                for dm in decision_markers:
                    dec = dm.get("decision", "CONTINUE")
                    color, icon = dec_icons.get(dec, ("blue", "question-sign"))
                    folium.Marker(
                        dm["pos"],
                        popup=f"<b>{dec}</b><br>{dm.get('rationale', '')[:100]}",
                        icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
                    ).add_to(dec_group)
                dec_group.add_to(m)

            # ── Map plugins ──
            try:
                plugins.Fullscreen(position="topleft").add_to(m)
                plugins.MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
                plugins.MousePosition(position="bottomleft").add_to(m)
            except Exception:
                pass

            # ── Custom legend ──
            legend_html = """
            <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                        background:rgba(0,0,0,0.8);padding:12px 16px;border-radius:8px;
                        font-family:monospace;font-size:11px;color:white;
                        border:1px solid #444;">
                <b style="font-size:13px;">KAIROS ENGINE</b><br>
                <span style="color:#2ECC40;">&#9679;</span> Origin (Pokhara)<br>
                <span style="color:#FF4136;">&#9679;</span> Destination (Muktinath)<br>
                <span style="color:#FF851B;">&#9679;</span> Contingency LZ<br>
                <span style="color:#E63946;">&#9644;</span> Flight Corridor<br>
                <hr style="border-color:#555;margin:4px 0;">
                <span style="color:#2ECC40;">&#9644;</span> Low Risk
                <span style="color:#FFDC00;">&#9644;</span> Medium
                <span style="color:#FF4136;">&#9644;</span> High Risk
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            # ── Layer control ──
            folium.LayerControl(collapsed=False).add_to(m)

            m.save(output_filename)
            logger.info(f"Flight map saved: {output_filename}")
            return output_filename

        except Exception as exc:
            logger.error(f"Map rendering failed: {exc}")
            return ""

    @staticmethod
    def render_risk_overlay(m, route_segments: List[Dict],
                            risk_scores: List[float]):
        """Add risk-colored route segments to an existing Folium map."""
        try:
            import folium
            for seg, score in zip(route_segments, risk_scores):
                color = _risk_color(score)
                folium.PolyLine(
                    [seg["from"], seg["to"]],
                    color=color, weight=5, opacity=0.9,
                    tooltip=f"Risk: {score:.0%}",
                ).add_to(m)
        except Exception as exc:
            logger.error(f"Risk overlay failed: {exc}")

    @staticmethod
    def render_decision_marker(m, position: List[float],
                                decision: str, rationale: str = ""):
        """Add a decision annotation marker to an existing Folium map."""
        try:
            import folium
            colors = {
                "CONTINUE": "green", "DIVERT": "orange", "LAND": "red",
                "ABORT": "darkred", "RTL": "purple",
            }
            folium.Marker(
                position,
                popup=f"<b>{decision}</b><br>{rationale[:150]}",
                icon=folium.Icon(color=colors.get(decision, "blue")),
            ).add_to(m)
        except Exception as exc:
            logger.error(f"Decision marker failed: {exc}")


def _risk_color(score: float) -> str:
    """Map risk score (0-1) to a hex color (green → yellow → red)."""
    if score < 0.3:
        return "#2ECC40"   # Green
    elif score < 0.5:
        return "#FFDC00"   # Yellow
    elif score < 0.7:
        return "#FF851B"   # Orange
    else:
        return "#FF4136"   # Red
