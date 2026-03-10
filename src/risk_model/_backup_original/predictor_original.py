"""
BACKUP - Original predictor.py
Restore: copy to ../predictor.py
"""

from typing import Any

from .finetuned_triage_predictor import FinetunedTriagePredictor
from .synapse_predictor import SynapseTriagePredictor


class RiskPredictor:
    def __init__(self, use_finetuned_triage: bool = True, use_synapse: bool = True):
        self._finetuned_triage = FinetunedTriagePredictor() if use_finetuned_triage else None
        self._synapse = SynapseTriagePredictor() if use_synapse else None

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if self._finetuned_triage and self._finetuned_triage.is_available:
            out = self._finetuned_triage.predict(features)
            if out is not None:
                return out
        if self._synapse and self._synapse.is_available:
            out = self._synapse.predict(features)
            if out is not None:
                return out
        return self._heuristic_predict(features)

    def _heuristic_predict(self, features: dict[str, Any]) -> dict[str, Any]:
        severity = "MODERATE"
        risk_score = 0.5
        if features.get("has_severe") or (features.get("severities") and features["severities"][0] == "severe"):
            severity = "HIGH"
            risk_score = 0.8
        elif features.get("severities") and features["severities"][0] == "mild":
            severity = "LOW"
            risk_score = 0.3

        if features.get("has_red_flag"):
            risk_score = min(1.0, risk_score + 0.15)
            if features.get("red_flag_severe"):
                severity, risk_score = "HIGH", 0.9
        if features.get("syndrome_alarm"):
            severity, risk_score = "HIGH", max(risk_score, 0.9)
        if features.get("age_group", 0) >= 3:
            risk_score = min(1.0, risk_score + 0.05)
        if features.get("comorbidity_count", 0) >= 2:
            risk_score = min(1.0, risk_score + 0.05)

        out = {
            "RiskScore": round(risk_score, 2),
            "Severity": severity,
            "Confidence": 0.7,
            "possible_conditions": self._possible_conditions(features),
        }
        out.setdefault("triage_recommendation", None)
        return out

    def _possible_conditions(self, features: dict[str, Any]) -> list[str]:
        conditions = []
        if features.get("syndrome_respiratory"):
            conditions.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            conditions.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            conditions.append("Gastrointestinal condition")
        return conditions
