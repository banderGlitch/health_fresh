"""
Stage 4: Risk Model — Predict triage (OTC | Doctor | Emergency).

Tries in order:
  1. Finetuned triage (3-way) — if model loaded (OTC Drug, Doctor Consultation, Emergency)
  2. SYNAPSE (2-way) — if model loaded (OTC Drug, Doctor Consultation)
  3. Heuristic — fallback using severity, red flags, syndromes  manual way 
"""

from typing import Any

from .finetuned_triage_predictor import FinetunedTriagePredictor
from .synapse_predictor import SynapseTriagePredictor


class RiskPredictor:
    """Predict risk score and triage from features."""

    def __init__(self, use_finetuned_triage: bool = True, use_synapse: bool = True):
        self._finetuned = FinetunedTriagePredictor() if use_finetuned_triage else None
        self._synapse = SynapseTriagePredictor() if use_synapse else None

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Try finetuned first, then SYNAPSE, then heuristic."""
        # Try finetuned triage
        if self._finetuned and self._finetuned.is_available:
            result = self._finetuned.predict(features)
            if result:
                return result

        # Try SYNAPSE
        if self._synapse and self._synapse.is_available:
            result = self._synapse.predict(features)
            if result:
                return result

        # Fallback: heuristic
        # return self._heuristic(features)

    # def _heuristic(self, features: dict[str, Any]) -> dict[str, Any]:
    #     """Simple rules when no model available."""
    #     severity = "MODERATE"
    #     risk = 0.5

    #     # From symptom severity
    #     if features.get("has_severe") or self._worst_severity(features) == "severe":
    #         severity, risk = "HIGH", 0.8
    #     elif self._worst_severity(features) == "mild":
    #         severity, risk = "LOW", 0.3

    #     # Red flags boost risk
    #     if features.get("has_red_flag"):
    #         risk = min(1.0, risk + 0.15)
    #         if features.get("red_flag_severe"):
    #             severity, risk = "HIGH", 0.9

    #     # Alarm (blood, etc.)
    #     if features.get("syndrome_alarm"):
    #         severity, risk = "HIGH", max(risk, 0.9)

    #     # Age 60+ and comorbidities
    #     if features.get("age_group", 0) >= 3:
    #         risk = min(1.0, risk + 0.05)
    #     if features.get("comorbidity_count", 0) >= 2:
    #         risk = min(1.0, risk + 0.05)

    #     return {
    #         "RiskScore": round(risk, 2),
    #         "Severity": severity,
    #         "Confidence": 0.7,
    #         "possible_conditions": self._conditions(features),
    #         "triage_recommendation": None,
    #     }

    # def _worst_severity(self, features: dict[str, Any]) -> str | None:
    #     """Get worst severity from features."""
    #     sev = features.get("severities") or []
    #     return sev[0] if sev else None

    # def _conditions(self, features: dict[str, Any]) -> list[str]:
    #     """Map syndromes to possible conditions."""
    #     out = []
    #     if features.get("syndrome_respiratory"):
    #         out.append("Respiratory infection (consider flu, COVID, pneumonia)")
    #     if features.get("syndrome_cardiac_like"):
    #         out.append("Cardiac or pulmonary consideration")
    #     if features.get("syndrome_gi"):
    #         out.append("Gastrointestinal condition")
    #     return out
