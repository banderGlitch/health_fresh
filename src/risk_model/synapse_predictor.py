"""
SYNAPSE Triage — 2-way output (OTC | Doctor).

Uses models/synapse_triage/ (RandomForest + TF-IDF).
Input: symptom text + demographics.
"""

from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "synapse_triage"


def _encode_age(age: int) -> int:
    """Age -> SYNAPSE code (0-5)."""
    if age < 5:
        return 5
    if age <= 15:
        return 2
    if age <= 45:
        return 0
    if age <= 60:
        return 1
    return 4


def _encode_gender(gender: str) -> int:
    """Gender -> 0=Female, 1=Male."""
    return 1 if str(gender).lower().strip() in ("male", "m") else 0


def _encode_duration(max_days: float | None, acute: int) -> int:
    """Duration -> 0=long, 1=short."""
    if acute:
        return 1
    if max_days is None:
        return 0
    return 1 if max_days <= 3 else 0


def _encode_severity(severities: list) -> int:
    """Worst severity -> 0=mild, 1=moderate, 2=severe."""
    m = {"mild": 0, "moderate": 1, "severe": 2}
    best = 0
    for s in (severities or []):
        if s:
            best = max(best, m.get(str(s).lower(), 1))
    return best


class SynapseTriagePredictor:
    """Predict triage using SYNAPSE model."""

    def __init__(self, model_dir: Path | str | None = None):
        self._dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._model = None
        self._vectorizer = None
        self._scaler = None
        self._label_enc = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            import joblib
            from scipy.sparse import hstack, csr_matrix
            self._model = joblib.load(self._dir / "telehealth_model.pkl")
            self._vectorizer = joblib.load(self._dir / "vectorizer.pkl")
            self._scaler = joblib.load(self._dir / "scaler.pkl")
            self._label_enc = joblib.load(self._dir / "label_encoder.pkl")
            self._hstack = hstack
            self._csr = csr_matrix
            self._loaded = True
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._loaded

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict triage from features."""
        if not self._loaded:
            return self._fallback(features)

        # Get inputs
        text = features.get("symptom_text") or "general discomfort"
        age = features.get("age", 40)
        gender = features.get("gender", "unknown")
        severities = features.get("severities") or []
        max_days = features.get("max_duration_days")
        acute = features.get("acute_flag", 0)

        # Encode
        import numpy as np
        text_feat = self._vectorizer.transform([text])
        num = np.array([[
            _encode_gender(gender),
            _encode_age(age),
            _encode_duration(max_days, acute),
            _encode_severity(severities),
        ]])
        num_scaled = self._scaler.transform(num)
        X = self._hstack([text_feat, self._csr(num_scaled)])

        # Predict
        pred = self._label_enc.inverse_transform(self._model.predict(X))[0]

        # Map to risk
        risk = 0.3 if pred == "OTC Drug" else 0.8
        severity = "LOW" if pred == "OTC Drug" else "HIGH"

        if features.get("has_red_flag") and features.get("red_flag_severe"):
            risk, severity = 0.95, "HIGH"

        return {
            "RiskScore": round(risk, 2),
            "Severity": severity,
            "Confidence": 0.85,
            "possible_conditions": self._conditions(features),
            "triage_recommendation": pred,
        }

    def _fallback(self, features: dict[str, Any]) -> dict[str, Any]:
        """When model not loaded."""
        severity, risk = "MODERATE", 0.5
        if features.get("has_severe"):
            severity, risk = "HIGH", 0.8
        elif self._worst_sev(features) == "mild":
            severity, risk = "LOW", 0.3
        if features.get("has_red_flag"):
            risk = min(1.0, risk + 0.15)
        return {
            "RiskScore": round(risk, 2),
            "Severity": severity,
            "Confidence": 0.5,
            "possible_conditions": [],
            "triage_recommendation": "Unknown (fallback)",
        }

    def _worst_sev(self, f: dict) -> str | None:
        sev = f.get("severities") or []
        return sev[0] if sev else None

    def _conditions(self, features: dict[str, Any]) -> list[str]:
        out = []
        if features.get("syndrome_respiratory"):
            out.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            out.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            out.append("Gastrointestinal condition")
        return out
