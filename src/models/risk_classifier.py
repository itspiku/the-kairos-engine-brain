"""
The Kairos Engine - XGBoost Crash Risk Classifier Model Wrapper

Singleton pattern with lazy loading, support for both 4-feature (legacy) and
8-feature (expanded) models, prediction explanation, and confidence intervals.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.config import RISK_MODEL_PATH, CRITICAL_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.core.exceptions import ModelLoadError

logger = logging.getLogger("kairos.risk")


class KairosRiskClassifier:
    """
    XGBoost flight risk prediction with singleton caching,
    feature explanation, and confidence intervals.
    """

    _instance: Optional["KairosRiskClassifier"] = None
    _FEATURE_NAMES_8 = [
        "battery_pct", "wind_speed_ms", "altitude_m", "action",
        "temperature_c", "visibility_km", "slope_gradient_deg", "icing_risk",
    ]
    _FEATURE_NAMES_4 = ["battery_pct", "wind_speed_ms", "altitude_m", "action"]

    def __new__(cls, model_path: str = RISK_MODEL_PATH):
        """Singleton pattern — reuse the same instance across calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = RISK_MODEL_PATH):
        if self._initialized:
            return
        self.model_path = model_path
        self.model = None
        self._n_features = None
        self._initialized = True

    def _ensure_loaded(self):
        """Lazy-load the XGBoost model from disk, training if missing."""
        if self.model is not None:
            return

        try:
            import xgboost as xgb
        except ImportError:
            raise ModelLoadError(self.model_path, reason="xgboost package not installed")

        if os.path.exists(self.model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                self._n_features = self.model.n_features_in_
                logger.info(f"Risk model loaded: {self.model_path} ({self._n_features} features)")
            except Exception as exc:
                raise ModelLoadError(self.model_path, reason=str(exc)) from exc
        else:
            logger.info("Risk model not found, training from synthetic data...")
            try:
                from build_crash_predictor import generate_synthetic_data, train_crash_predictor
                df = generate_synthetic_data()
                self.model = train_crash_predictor(df, model_path=self.model_path)
                self._n_features = self.model.n_features_in_
            except Exception as exc:
                raise ModelLoadError(self.model_path, reason=f"Auto-training failed: {exc}") from exc

    def predict_risk(self, battery_pct: float, wind_speed_ms: float,
                     altitude_m: float, proposed_action: str,
                     temperature_c: float = 5.0, visibility_km: float = 10.0,
                     slope_gradient_deg: float = 10.0,
                     icing_risk: float = 0.0) -> Dict[str, Any]:
        """
        Predict crash probability and risk level.
        Supports both 4-feature and 8-feature models automatically.
        """
        self._ensure_loaded()
        import numpy as np

        action_map = {"CONTINUE": 0, "RTL": 1, "DIVERT": 2, "LAND": 2, "ABORT": 1}
        action_code = action_map.get(str(proposed_action).upper(), 0)

        # Build feature vector based on model's expected features
        if self._n_features and self._n_features >= 8:
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

        probs = self.model.predict_proba(features)[0]
        crash_prob = float(probs[1])

        if crash_prob > CRITICAL_RISK_THRESHOLD:
            risk_level = "Critical"
        elif crash_prob > HIGH_RISK_THRESHOLD:
            risk_level = "High"
        else:
            risk_level = "Acceptable"

        result = {
            "crash_probability": round(crash_prob, 2),
            "risk_level": risk_level,
            "data_source": f"XGBoost ({self._n_features or 4}-feature Himalayan BVLOS Corpus)",
        }

        return result

    def explain_prediction(self, battery_pct: float, wind_speed_ms: float,
                           altitude_m: float, proposed_action: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Return feature contributions for a prediction (basic importance-based explanation).
        """
        self._ensure_loaded()

        prediction = self.predict_risk(battery_pct, wind_speed_ms, altitude_m, proposed_action, **kwargs)

        # Use model feature importances as approximate explanations
        try:
            importances = self.model.feature_importances_
            if self._n_features and self._n_features >= 8:
                names = self._FEATURE_NAMES_8
            else:
                names = self._FEATURE_NAMES_4

            contributions = {}
            for name, imp in zip(names, importances):
                contributions[name] = round(float(imp), 4)

            prediction["feature_contributions"] = dict(
                sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            )
        except Exception:
            prediction["feature_contributions"] = {}

        return prediction

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
