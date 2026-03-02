"""
Finetuned Triage Predictor — fine-tuned SYNAPSE risk model with 3-way output.

Uses models/finetuned_triage/ (risk_model.pkl, label_encoders.pkl).
Input: 5 features (symptom category, gender, age bucket, duration, severity)
Output: OTC Drug | Doctor Consultation | Emergency
"""

from pathlib import Path
from typing import Any

from .triage_mapper import features_to_triage_input

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "models" / "finetuned_triage"


class FinetunedTriagePredictor:
    """
    Risk predictor backed by fine-tuned triage model.
    3-way output: OTC Drug | Doctor Consultation | Emergency.
    """

    def __init__(self, model_dir: Path | str | None = None):
        self._model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._model = None
        self._label_encoders = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            import joblib
            import numpy as np

            le_path = self._model_dir / "label_encoders.pkl"
            rm_path = self._model_dir / "risk_model.pkl"
            if not le_path.exists() or not rm_path.exists():
                self._loaded = False
                return

            self._label_encoders = joblib.load(le_path)
            self._model = joblib.load(rm_path)
            self._np = np
            self._loaded = True
        except Exception:
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Run triage prediction.
        Maps pipeline features to model input, runs model.
        Returns RiskScore, Severity, triage_recommendation (OTC|Doctor|Emergency).
        """
        if not self._loaded:
            return None  # Caller should fallback

        triage_input = features_to_triage_input(features)
        if not triage_input:
            return None

        # Encode and predict
        le = self._label_encoders
        symptom_enc = le["Symptoms"].transform([triage_input["symptom_category"]])[0]
        gender_enc = le["Gender"].transform([triage_input["gender"]])[0]
        age_enc = le["Age"].transform([triage_input["age_bucket"]])[0]
        duration_enc = le["Duration"].transform([triage_input["duration"]])[0]
        severity_enc = le["Severity"].transform([triage_input["severity"]])[0]

        X = self._np.array(
            [[symptom_enc, gender_enc, age_enc, duration_enc, severity_enc]]
        )
        pred_enc = self._model.predict(X)[0]
        triage = le["Final Recommendation"].inverse_transform([pred_enc])[0]

        # Map to RiskScore and Severity
        if triage == "OTC Drug":
            risk_score = 0.3
            severity = "LOW"
        elif triage == "Emergency":
            risk_score = 0.95
            severity = "HIGH"
        else:
            risk_score = 0.8
            severity = "HIGH"

        # Red-flag override (safety)
        if features.get("has_red_flag") and features.get("red_flag_severe"):
            risk_score = 0.95
            severity = "HIGH"
            triage = "Emergency"

        return {
            "RiskScore": round(risk_score, 2),
            "Severity": severity,
            "Confidence": 0.85,
            "possible_conditions": self._possible_conditions(features),
            "triage_recommendation": triage,
        }

    def _possible_conditions(self, features: dict[str, Any]) -> list[str]:
        conditions = []
        if features.get("syndrome_respiratory"):
            conditions.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            conditions.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            conditions.append("Gastrointestinal condition")
        return conditions
