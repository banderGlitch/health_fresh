"""
Finetuned Triage — 3-way output (OTC | Doctor | Emergency).

Uses models/finetuned_triage/ (risk_model.pkl, label_encoders.pkl).
Input: 5 features from triage_mapper.
"""

from pathlib import Path
from typing import Any

from .triage_mapper import features_to_triage_input

# Paths
_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "finetuned_triage"

# Triage -> risk score mapping
_RISK_MAP = {"OTC Drug": (0.3, "LOW"), "Emergency": (0.95, "HIGH"), "Doctor Consultation": (0.8, "HIGH")}


class FinetunedTriagePredictor:
    """Predict triage using finetuned model."""

    def __init__(self, model_dir: Path | str | None = None):
        self._dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._model = None
        self._encoders = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load model and encoders."""
        try:
            import joblib
            import numpy as np
            le_path = self._dir / "label_encoders.pkl"
            rm_path = self._dir / "risk_model.pkl"
            if not le_path.exists() or not rm_path.exists():
                return
            self._encoders = joblib.load(le_path)
            self._model = joblib.load(rm_path)
            self._np = np
            self._loaded = True
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._loaded

    def predict(self, features: dict[str, Any]) -> dict[str, Any] | None:
        """Predict triage from features."""
        if not self._loaded:
            return None

        # Convert features to model input
        inp = features_to_triage_input(features)
        if not inp:
            return None

        # Encode and predict
        le = self._encoders
        row = [
            le["Symptoms"].transform([inp["symptom_category"]])[0],
            le["Gender"].transform([inp["gender"]])[0],
            le["Age"].transform([inp["age_bucket"]])[0],
            le["Duration"].transform([inp["duration"]])[0],
            le["Severity"].transform([inp["severity"]])[0],
        ]

#         inp = {
#  "symptom_category": "Respiratory",
#  "gender": "Male",
#  "age_bucket": "40-60",
#  "duration": "Short",
#  "severity": "Severe"
# }

# encoding 
# Respiratory → 2
# Cardiac → 1
# GI → 0

# Respiratory → 2
# Cardiac → 1
# GI → 0

        X = self._np.array([row])
        # The label encoder converts the number back into text.
        pred = le["Final Recommendation"].inverse_transform(self._model.predict(X))[0]
        # 2 → "Emergency"
        # 1 → "Doctor Consultation"
        # 0 → "OTC Drug"

        # Map to risk score
        risk, severity = _RISK_MAP.get(pred, (0.8, "HIGH"))

        # Safety: red-flag + severe -> Emergency
        if features.get("has_red_flag") and features.get("red_flag_severe"):
            risk, severity, pred = 0.95, "HIGH", "Emergency"

        return {
            "RiskScore": round(risk, 2),
            "Severity": severity,
            "Confidence": 0.85,  # hard coded confidence
            "possible_conditions": self._conditions(features),
            "triage_recommendation": pred,
        }

    def _conditions(self, features: dict[str, Any]) -> list[str]:
        out = []
        if features.get("syndrome_respiratory"):
            out.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            out.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            out.append("Gastrointestinal condition")
        return out
