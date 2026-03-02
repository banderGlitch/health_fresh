"""
Stage 3: Feature Builder

Builds feature dict from symptoms, demographics, history.
Input for Risk Model. See docs/RISK_MODEL_FEATURES.md for full spec.

Output includes:
  - symptom_text: comma-separated for SYNAPSE TF-IDF
  - symptom_count, has_severe, max_severity
  - has_red_flag, syndrome_* (cardiac, respiratory, gi, alarm)
  - age, age_group, gender
  - comorbidity flags (diabetes, hypertension, etc.)
"""

import re
from typing import Any

from .risk_features import RED_FLAG_SYMPTOMS, SEVERITY_ENCODING, SYNDROME_DEFINITIONS


class FeatureBuilder:
    """
    Builds features for risk model:
    - Symptom presence, counts, severity
    - Duration / temporal
    - Red-flag and syndrome indicators
    - Demographics, medical history
    """

    def build(
        self,
        mapped_symptoms: list[dict[str, Any]],
        extraction_dict: dict[str, Any] | None = None,
        demographics: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build feature dict for risk model. See RISK_MODEL_FEATURES.md."""
        demographics = demographics or {}
        history = history or {}

        # Prefer extraction_dict (from NER) so we include unmapped symptoms
        all_symptoms = []
        if extraction_dict and extraction_dict.get("symptoms"):
            for s in extraction_dict["symptoms"]:
                all_symptoms.append({
                    "name": s.get("name", ""),
                    "canonical_name": s.get("name", ""),
                    "duration": s.get("duration"),
                    "severity": s.get("severity"),
                })
        else:
            all_symptoms = mapped_symptoms

        # Symptom presence: names, text for SYNAPSE, SNOMED codes
        symptom_names = [str(s.get("canonical_name", s.get("name", ""))).lower() for s in all_symptoms]
        symptom_names_set = {n.strip() for n in symptom_names if n}
        symptom_text = ", ".join(n for n in symptom_names if n.strip()) or "general discomfort"
        symptom_codes = [s.get("snomed_code") for s in mapped_symptoms if s.get("snomed_code")]
        severities = [s.get("severity") for s in all_symptoms]
        durations = [s.get("duration") for s in all_symptoms]

        symptom_count = len(all_symptoms)

        # --- Severity features ---
        sev_encoded = [SEVERITY_ENCODING.get(s, 0.5) for s in severities if s]
        max_severity = max(sev_encoded) if sev_encoded else 0.5
        severe_count = sum(1 for s in severities if s == "severe")
        has_severe = 1 if severe_count > 0 else 0
        severity_weighted_sum = sum(sev_encoded)

        # Duration: parse to days, acute (<24h), chronic (>2 weeks)
        max_duration_days = self._parse_max_duration_days(durations)
        acute_flag = 1 if self._has_acute(durations) else 0
        chronic_flag = 1 if max_duration_days and max_duration_days > 14 else 0

        # Red-flag: chest pain, shortness of breath, coughing blood, blood in stool
        red_flag_count = sum(1 for n in symptom_names_set if n in RED_FLAG_SYMPTOMS)
        has_red_flag = 1 if red_flag_count > 0 else 0
        has_severe_red_flag = has_red_flag and has_severe

        # Syndromes: cardiac-like, respiratory, GI, alarm (from risk_features.SYNDROME_DEFINITIONS)
        syndromes = {}
        for name, symptom_set in SYNDROME_DEFINITIONS.items():
            syndromes[f"syndrome_{name}"] = 1 if symptom_set.intersection(symptom_names_set) else 0

        # Demographics: age (default 40), age_group (0-3), gender
        age = self._parse_age(demographics.get("age"))
        age_group = self._age_group(age)
        gender = str(demographics.get("gender", "")).lower() or "unknown"

        # Medical history: diabetes, hypertension, heart, lung -> binary flags
        has_diabetes = 1 if _history_has(demographics, history, "diabetes") else 0
        has_hypertension = 1 if _history_has(demographics, history, "hypertension") else 0
        has_heart_disease = 1 if _history_has(demographics, history, "heart") else 0
        has_lung_disease = 1 if _history_has(demographics, history, "lung") else 0
        comorbidity_count = has_diabetes + has_hypertension + has_heart_disease + has_lung_disease

        return {
            # Legacy (for current predictor)
            "symptom_codes": symptom_codes,
            "symptom_vector": list(set(symptom_codes)),
            "symptom_text": symptom_text,  # For SYNAPSE: comma-separated names
            "demographics": demographics,
            "history": history,
            "severities": severities,
            "durations": durations,
            # New: structured features for ML
            "symptom_count": symptom_count,
            "max_severity": max_severity,
            "has_severe": has_severe,
            "severe_count": severe_count,
            "severity_weighted_sum": severity_weighted_sum,
            "max_duration_days": max_duration_days,
            "acute_flag": acute_flag,
            "chronic_flag": chronic_flag,
            "has_red_flag": has_red_flag,
            "red_flag_count": red_flag_count,
            "red_flag_severe": 1 if has_severe_red_flag else 0,
            **syndromes,
            "age": age,
            "age_group": age_group,
            "gender": gender,
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
            "has_heart_disease": has_heart_disease,
            "has_lung_disease": has_lung_disease,
            "comorbidity_count": comorbidity_count,
        }

    def _parse_max_duration_days(self, durations: list[str | None]) -> float | None:
        """Parse duration strings to days. Returns max, or None if none parseable."""
        days_list = []
        for d in durations:
            if not d:
                continue
            days_list.append(self._parse_duration_to_days(d))
        valid = [x for x in days_list if x is not None]
        return max(valid) if valid else None

    def _parse_duration_to_days(self, s: str) -> float | None:
        """Parse '3 days', '2 weeks', 'a week' etc. to days."""
        s = s.lower().strip()
        m = re.search(r"(\d+)\s*(day|week|month|hour)", s)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit.startswith("day"):
                return float(n)
            if unit.startswith("week"):
                return n * 7.0
            if unit.startswith("month"):
                return n * 30.0
            if unit.startswith("hour"):
                return n / 24.0
        if "week" in s or "a week" in s:
            return 7.0
        if "day" in s or "few days" in s:
            return 3.0
        if "month" in s:
            return 30.0
        return None

    def _has_acute(self, durations: list[str | None]) -> bool:
        """True if any duration suggests < 24h (e.g. 'few hours')."""
        for d in durations:
            if not d:
                continue
            d = d.lower()
            if "hour" in d and "day" not in d:
                return True
            if "today" in d or "yesterday" in d:
                return True
        return False

    def _parse_age(self, age: Any) -> int:
        """Parse age. Default 40 if missing/invalid."""
        if age is None:
            return 40
        try:
            n = int(age)
            return max(0, min(120, n))
        except (TypeError, ValueError):
            return 40

    def _age_group(self, age: int) -> int:
        """0=<18, 1=18-40, 2=40-60, 3=60+."""
        if age < 18:
            return 0
        if age < 40:
            return 1
        if age < 60:
            return 2
        return 3


def _history_has(demographics: dict, history: dict, keyword: str) -> bool:
    """Check if history or demographics contain keyword (e.g. diabetes, heart)."""
    kw = keyword.lower()
    for d in (demographics, history):
        for k, v in (d or {}).items():
            if kw in str(k).lower() and v:
                return True
            if isinstance(v, str) and kw in v.lower():
                return True
            if isinstance(v, dict) and any(kw in str(x).lower() for x in v):
                return True
    return False
