"""
The Kairos Engine - Unit Normalization at Module Boundaries

The engine speaks percent for battery level (TelemetryReport.battery_pct == 42.0
means 42%), but the XGBoost risk model was trained on fractions in [0.10, 0.60].
Both conventions reach the risk tools: telemetry and the LLM prompt carry percent,
while the tool schema historically documented fractions. Converting at the boundary
keeps each side speaking its own units without a silent mismatch in between.
"""

from typing import Union

Number = Union[int, float]


def normalize_battery_fraction(value: Number) -> float:
    """
    Coerce a battery level to a 0.0-1.0 fraction.

    Values above 1.0 are read as percent (42 -> 0.42, 100 -> 1.0). Values of 1.0
    or below are already fractions and pass through unchanged (0.42 -> 0.42).

    The function is **idempotent**, which is the property that matters: it is
    applied both at the tool boundary and inside the classifier, and a rule that
    re-scaled an already-normalized value would turn a full pack (100 -> 1.0)
    into 1% on the second pass — the exact failure this module exists to prevent.

    The cost of idempotency is that a literal 1.0 is read as 100%, not 1%. That
    residual is acceptable: the engine's own convention delivers a full battery
    as 100.0 rather than 1.0, and a genuine 1% pack trips the anomaly detector's
    BATTERY_CRIT threshold (15%) independently of the risk model.
    """
    v = float(value)
    if v != v:  # NaN
        raise ValueError("battery level is NaN")
    if v < 0.0:
        raise ValueError(f"battery level is negative: {v}")
    if v > 1.0:
        v = v / 100.0
    return min(v, 1.0)


def battery_fraction_to_pct(fraction: Number) -> float:
    """Convert a 0.0-1.0 fraction back to percent for display and telemetry."""
    return round(float(fraction) * 100.0, 1)
