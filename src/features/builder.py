"""
Stage 3: Feature Builder
Constructs feature vectors for the risk model.
"""

from typing import Any


class FeatureBuilder:
    """
    Builds features from:
    - Symptom vectors (ontology-mapped)
    - Demographics (age, gender, etc.)
    - History (medical history, temporal patterns)
    """

    def build(
        self,
        mapped_symptoms: list[dict[str, Any]],
        demographics: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build feature dict for risk model input."""
        demographics = demographics or {}
        history = history or {}

        # Symptom vector: one-hot or embedding of SNOMED codes
        symptom_codes = [s.get("snomed_code") for s in mapped_symptoms if s.get("snomed_code")]
        symptom_vector = list(set(symptom_codes))  # Simplified; use proper encoding in production

        return {
            "symptom_codes": symptom_codes,
            "symptom_vector": symptom_vector,
            "demographics": demographics,
            "history": history,
            "severities": [s.get("severity") for s in mapped_symptoms],
            "durations": [s.get("duration") for s in mapped_symptoms],
        }
