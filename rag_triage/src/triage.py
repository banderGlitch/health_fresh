"""RAG-based triage predictor — replaces traditional ML risk model."""

from typing import Any

from config import TOP_K
from .retriever import Retriever


def _age_to_synapse(age: int | None) -> str:
    """Map numeric age to SYNAPSE age bucket."""
    if age is None or age < 0:
        return "16-60 years"
    if age < 5:
        return "below 5 years"
    if age <= 15:
        return "6-15 years"
    if age <= 45:
        return "16-60 years"
    if age <= 60:
        return "16-60 years"
    return "above 45 years"


def _duration_to_synapse(duration_days: float | None, acute: bool = False) -> str:
    """Map duration to SYNAPSE format."""
    if acute:
        return "Less than 3 days"
    if duration_days is None:
        return "Greater than 3 days"
    return "Less than 3 days" if duration_days <= 3 else "Greater than 3 days"


def _severity_to_synapse(severity: str | None) -> str:
    """Map severity to SYNAPSE format (Mild, Moderate, Severe)."""
    if not severity:
        return "Moderate"
    s = str(severity).strip().lower()
    if s in ("mild", "low"):
        return "Mild"
    if s in ("severe", "high"):
        return "Severe"
    return "Moderate"


def _build_query(
    symptom_text: str,
    age: int | None = None,
    gender: str | None = None,
    duration_days: float | None = None,
    severity: str | None = None,
    acute: bool = False,
) -> str:
    """Build query text in SYNAPSE format for retrieval."""
    parts = [str(symptom_text or "general discomfort").strip()]
    parts.append(str(gender or "unknown").strip())
    parts.append(_age_to_synapse(age))
    parts.append(_duration_to_synapse(duration_days, acute))
    parts.append(_severity_to_synapse(severity))
    return " | ".join(parts)


class RAGTriagePredictor:
    """
    Predict triage using RAG: retrieve similar SYNAPSE cases, majority vote.
    Drop-in replacement for traditional ML risk model.
    """

    def __init__(self, top_k: int | None = None):
        self._retriever = Retriever()
        self._top_k = top_k or TOP_K

    @property
    def is_available(self) -> bool:
        try:
            self._retriever._load()
            return True
        except FileNotFoundError:
            return False

    def predict(
        self,
        features: dict[str, Any] | None = None,
        *,
        symptom_text: str | None = None,
        age: int | None = None,
        gender: str | None = None,
        duration_days: float | None = None,
        severity: str | None = None,
        acute: bool = False,
    ) -> dict[str, Any]:
        """
        Predict triage from features dict (pipeline compatibility) or explicit kwargs.
        Returns: RiskScore, Severity, Confidence, triage_recommendation, similar_cases, possible_conditions.
        """
        if features:
            symptom_text = features.get("symptom_text") or symptom_text or "general discomfort"
            age = features.get("age") if age is None else age
            gender = features.get("gender") if gender is None else gender
            duration_days = features.get("max_duration_days") if duration_days is None else duration_days
            acute = features.get("acute_flag", 0) or acute
            sevs = features.get("severities") or []
            severity = sevs[0] if sevs else severity
        else:
            symptom_text = symptom_text or "general discomfort"

        query = _build_query(symptom_text, age, gender, duration_days, severity, acute)
        results = self._retriever.retrieve(query, top_k=self._top_k)

        if not results:
            return self._fallback(symptom_text)

        # Majority vote on recommendation
        rec_counts: dict[str, int] = {}
        for case, score in results:
            rec = case.get("recommendation", "")
            if rec:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1

        if not rec_counts:
            return self._fallback(symptom_text)

        winner = max(rec_counts, key=rec_counts.get)
        total = sum(rec_counts.values())
        confidence = rec_counts[winner] / total if total else 0.5

        risk = 0.3 if winner == "OTC Drug" else 0.8
        sev_label = "LOW" if winner == "OTC Drug" else "HIGH"

        similar_cases = [
            {"symptoms": c["symptoms"], "recommendation": c["recommendation"], "score": round(s, 4)}
            for c, s in results[:5]
        ]

        possible_conditions = self._infer_conditions(symptom_text, results)

        return {
            "RiskScore": round(risk, 2),
            "Severity": sev_label,
            "Confidence": round(confidence, 2),
            "triage_recommendation": winner,
            "similar_cases": similar_cases,
            "possible_conditions": possible_conditions,
        }

    def _fallback(self, symptom_text: str) -> dict[str, Any]:
        """When retrieval fails."""
        return {
            "RiskScore": 0.5,
            "Severity": "MODERATE",
            "Confidence": 0.5,
            "triage_recommendation": "Unknown (RAG index not built)",
            "similar_cases": [],
            "possible_conditions": [],
        }

    def _infer_conditions(self, symptom_text: str, results: list) -> list[str]:
        """Infer possible conditions from retrieved cases (Phase 1: simple heuristics)."""
        out = []
        text_lower = (symptom_text or "").lower()
        if any(w in text_lower for w in ["cough", "fever", "breath", "throat"]):
            out.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if any(w in text_lower for w in ["chest", "heart", "breath"]):
            out.append("Cardiac or pulmonary consideration")
        if any(w in text_lower for w in ["stomach", "abdominal", "diarrhea", "vomit"]):
            out.append("Gastrointestinal condition")
        return out[:3]
