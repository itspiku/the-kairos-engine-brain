"""
XGBoost Crash Predictor — Advanced ML Pipeline for The Kairos Engine

Generates 10,000 physics-informed Himalayan flight scenarios with 8 features,
trains an XGBoost binary classifier with 5-fold cross-validation, outputs
confusion matrix, precision/recall/F1, and feature importance ranking.
"""

import sys

import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


# ── STEP 1: Generate Physics-Informed Synthetic Data (8 Features) ─────────
def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic flight telemetry data with 8 features and
    5 physics-informed Himalayan crash rules.
    """
    np.random.seed(seed)

    battery_pct = np.random.uniform(0.10, 0.60, n_samples)
    wind_speed_ms = np.random.uniform(0.0, 25.0, n_samples)
    altitude_m = np.random.uniform(1000, 5000, n_samples)
    action = np.random.choice([0, 1, 2], n_samples)  # 0:CONTINUE, 1:RTL, 2:DIVERT
    temperature_c = np.random.uniform(-20, 25, n_samples)
    visibility_km = np.random.uniform(0.5, 30, n_samples)
    slope_gradient_deg = np.random.uniform(0, 45, n_samples)
    icing_risk = np.clip(((-temperature_c - 5) / 15) * (altitude_m / 5000), 0, 1)

    crash = []
    for b, w, a, act, temp, vis, slope, ice in zip(
        battery_pct, wind_speed_ms, altitude_m, action,
        temperature_c, visibility_km, slope_gradient_deg, icing_risk,
    ):
        # Rule 1 (Mountain RTL): high altitude + low battery + climbing over ridge
        if act == 1 and a > 3000 and b < 0.35:
            is_crash = 1 if np.random.random() < 0.90 else 0

        # Rule 2 (Headwind Exhaustion): high headwind + low battery + continuing
        elif act == 0 and w > 15 and b < 0.25:
            is_crash = 1 if np.random.random() < 0.85 else 0

        # Rule 3 (Safe Divert): divert with sufficient battery
        elif act == 2 and b > 0.15:
            is_crash = 0 if np.random.random() < 0.98 else 1

        # Rule 4 (Icing): extreme cold + high altitude + low visibility
        elif temp < -10 and a > 3500 and vis < 2.0:
            is_crash = 1 if np.random.random() < 0.80 else 0

        # Rule 5 (Steep Terrain): steep slope + moderate wind
        elif slope > 30 and w > 12:
            is_crash = 1 if np.random.random() < 0.75 else 0

        # Default: energy balance equation
        else:
            energy_required = (w * 0.05) + (a * 0.0001) + max(0, -temp * 0.002)
            is_crash = 1 if energy_required > b else 0

        crash.append(is_crash)

    return pd.DataFrame({
        "battery_pct": battery_pct,
        "wind_speed_ms": wind_speed_ms,
        "altitude_m": altitude_m,
        "action": action,
        "temperature_c": temperature_c,
        "visibility_km": visibility_km,
        "slope_gradient_deg": slope_gradient_deg,
        "icing_risk": icing_risk,
        "crash": crash,
    })


# ── STEP 2: Train and Evaluate XGBoost Model ─────────────────────────────
def train_crash_predictor(df: pd.DataFrame,
                          model_path: str = "kairos_crash_predictor.json") -> xgb.XGBClassifier:
    """
    Train XGBClassifier with 5-fold cross-validation and comprehensive metrics.
    Saves model with metadata.
    """
    feature_cols = [
        "battery_pct", "wind_speed_ms", "altitude_m", "action",
        "temperature_c", "visibility_km", "slope_gradient_deg", "icing_risk",
    ]
    X = df[feature_cols]
    y = df["crash"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n   Training Complete.")
    print(f"   Accuracy  : {acc * 100:.2f}%")
    print(f"   Precision : {prec * 100:.2f}%")
    print(f"   Recall    : {rec * 100:.2f}%")
    print(f"   F1 Score  : {f1 * 100:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"     TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"     FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")

    # ── 5-Fold Cross-Validation ──
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n   5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # ── Feature Importance ──
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"\n   Feature Importance Ranking:")
    for i, (feat, imp) in enumerate(ranked, 1):
        bar = "#" * int(imp * 40)
        print(f"     {i}. {feat:<22s} {imp:.4f} {bar}")

    # ── Save ──
    model.save_model(model_path)
    print(f"\n   Model saved -> {model_path}")

    # Save metadata alongside
    meta_path = model_path.replace(".json", "_metadata.json")
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved -> {meta_path}\n")

    return model


# ── STEP 3: Tool Wrapper for Gemma 4 ─────────────────────────────────────
def assess_risk(battery_pct: float, wind_speed_ms: float,
                altitude_m: int, proposed_action: str,
                temperature_c: float = 5.0, visibility_km: float = 10.0,
                slope_gradient_deg: float = 10.0, icing_risk: float = 0.0,
                model_path: str = "kairos_crash_predictor.json") -> dict:
    """
    Tool function called by Gemma 4 to assess flight crash risk.
    Supports both 4-feature (legacy) and 8-feature (expanded) calls.

    `battery_pct` accepts percent (42) or fraction (0.42); the model is trained
    on fractions, so it is normalized here.
    """
    # Imported lazily to keep this training script standalone.
    from src.core.units import normalize_battery_fraction

    battery_pct = normalize_battery_fraction(battery_pct)
    action_map = {"CONTINUE": 0, "RTL": 1, "DIVERT": 2, "LAND": 2, "ABORT": 1}
    action_code = action_map.get(str(proposed_action).upper(), 0)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    n_features = model.n_features_in_
    if n_features >= 8:
        features = np.array([[
            float(battery_pct), float(wind_speed_ms), float(altitude_m),
            int(action_code), float(temperature_c), float(visibility_km),
            float(slope_gradient_deg), float(icing_risk),
        ]])
    else:
        features = np.array([[
            float(battery_pct), float(wind_speed_ms),
            float(altitude_m), int(action_code),
        ]])

    probs = model.predict_proba(features)[0]
    crash_prob = float(probs[1])

    if crash_prob > 0.7:
        risk_level = "Critical"
    elif crash_prob > 0.4:
        risk_level = "High"
    else:
        risk_level = "Acceptable"

    return {
        "crash_probability": round(crash_prob, 2),
        "risk_level": risk_level,
        "data_source": f"{n_features}-feature Himalayan BVLOS Corpus (10,000 flight logs)",
    }


# ── STEP 4: CLI Entry ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  KAIROS ENGINE - XGBOOST CRASH PREDICTOR (8-FEATURE EXPANDED)")
    print("=" * 64)

    print("\n[1] Generating 10,000 physics-informed flight logs (8 features)...")
    data = generate_synthetic_data(n_samples=10000)
    print(f"   Crash rate: {data['crash'].mean()*100:.1f}%")

    print("\n[2] Training XGBoost Classifier with 5-fold CV...")
    trained_model = train_crash_predictor(data)

    print("[3] Testing `assess_risk` tool wrapper:")
    test_cases = [
        {"battery_pct": 0.2, "wind_speed_ms": 18, "altitude_m": 4000,
         "proposed_action": "RTL", "temperature_c": -12, "visibility_km": 1.5},
        {"battery_pct": 0.55, "wind_speed_ms": 5, "altitude_m": 1500,
         "proposed_action": "DIVERT", "temperature_c": 15, "visibility_km": 20},
        {"battery_pct": 0.15, "wind_speed_ms": 22, "altitude_m": 4500,
         "proposed_action": "CONTINUE", "temperature_c": -18, "visibility_km": 0.8},
    ]
    for params in test_cases:
        result = assess_risk(**params)
        action = params["proposed_action"]
        print(f"\n   {action} @ {params['battery_pct']*100:.0f}% battery, "
              f"{params['wind_speed_ms']}m/s wind:")
        print(f"   -> {json.dumps(result)}")

    print("\n" + "=" * 64)
