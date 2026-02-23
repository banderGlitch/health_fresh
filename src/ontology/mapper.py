"""
Stage 2: Ontology Mapping
Maps patient terms to standard medical concepts (e.g., SNOMED CT).
"""

from typing import Any


# Example SNOMED mappings - extend with full ontology in production
SNOMED_SYMPTOM_MAP = {
    "fever": ("386661006", "Fever"),
    "pyrexia": ("386661006", "Fever"),
    "chest pain": ("29857009", "Chest pain"),
    "headache": ("25064002", "Headache"),
    "nausea": ("422587007", "Nausea"),
    "vomiting": ("422400008", "Vomiting"),
    "shortness of breath": ("267036007", "Dyspnea"),
    "dyspnea": ("267036007", "Dyspnea"),
    "cough": ("49727002", "Cough"),
    "fatigue": ("84229001", "Fatigue"),
}


class OntologyMapper:
    """
    Maps extracted symptoms to standard medical ontology codes.
    TODO: Integrate full SNOMED CT / UMLS in production.
    """

    def map_symptom(self, symptom_name: str) -> dict[str, Any] | None:
        """Map a symptom to SNOMED code and canonical name."""
        key = symptom_name.lower().strip()
        if key in SNOMED_SYMPTOM_MAP:
            code, canonical = SNOMED_SYMPTOM_MAP[key]
            return {
                "snomed_code": code,
                "canonical_name": canonical,
                "original": symptom_name,
            }
        return None

    def map_extraction_result(self, extraction_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Map all symptoms from extraction stage to ontology."""
        mapped = []
        for s in extraction_output.get("symptoms", []):
            name = s.get("name", "")
            result = self.map_symptom(name)
            if result:
                result["duration"] = s.get("duration")
                result["severity"] = s.get("severity")
                mapped.append(result)
        return mapped
