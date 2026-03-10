"""
Stage 2: Ontology Mapping

Maps extracted symptoms to SNOMED CT codes.
Extend SNOMED_SYMPTOM_MAP for production use.
"""

from typing import Any

# SNOMED CT symptom mappings (code, canonical name)
# dynamic if we can fetch vai an api 
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
        """Map all symptoms from NER extraction to SNOMED. Unmapped symptoms are excluded from list."""
        mapped = []
        for s in extraction_output.get("symptoms", []):
            name = s.get("name", "")
            result = self.map_symptom(name)
            if result:
                # Preserve duration/severity from NER
                result["duration"] = s.get("duration")
                result["severity"] = s.get("severity")
                mapped.append(result)
        return mapped
