"""
Stage 3: Build feature dict for risk model from symptoms + demographics.
"""

import re
from typing import Any

from .risk_features import RED_FLAG_SYMPTOMS, SEVERITY_ENCODING, SYNDROME_DEFINITIONS


def _days(s: str | None) -> float | None:
    if not s:
        return None
    s = s.lower()
    m = re.search(r"(\d+)\s*(day|week|month|hour)", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        if u.startswith("day"): return float(n)
        if u.startswith("week"): return n * 7.0
        if u.startswith("month"): return n * 30.0
        if u.startswith("hour"): return n / 24.0
    if "week" in s: return 7.0
    if "day" in s or "few" in s: return 3.0
    if "month" in s: return 30.0
    return None


def _acute(durations: list) -> bool:
    for d in durations:
        if not d:
            continue
        d = d.lower()
        if ("hour" in d and "day" not in d) or "today" in d or "yesterday" in d:
            return True
    return False


def _age(v: Any) -> int:
    try:
        return max(0, min(120, int(v or 40)))
    except (TypeError, ValueError):
        return 40


def _age_group(a: int) -> int:
    return 0 if a < 18 else 1 if a < 40 else 2 if a < 60 else 3


def _has(d: dict, h: dict, w: str) -> bool:
    w = w.lower()
    for dct in (d or {}, h or {}):
        for k, v in dct.items():
            if v and (w in str(k).lower() or (isinstance(v, str) and w in v.lower())):
                return True
    return False


class FeatureBuilder:
    """Build feature dict for risk model."""

    def build(
        self,
        mapped_symptoms: list[dict],
        extraction_dict: dict | None = None,
        demographics: dict | None = None,
        history: dict | None = None,
    ) -> dict[str, Any]:
        dem, hist = demographics or {}, history or {}
        syms = (extraction_dict or {}).get("symptoms") or mapped_symptoms
        syms = [{"name": s.get("name", ""), "canonical_name": s.get("canonical_name", s.get("name", "")), "duration": s.get("duration"), "severity": s.get("severity")} for s in syms]

        names = [str(s.get("canonical_name", s.get("name", ""))).lower().strip() for s in syms if s.get("name") or s.get("canonical_name")]
        names_set = {n for n in names if n}
        text = ", ".join(names) or "general discomfort"
        codes = [s.get("snomed_code") for s in mapped_symptoms if s.get("snomed_code")]
        sevs = [s.get("severity") for s in syms]
        durs = [s.get("duration") for s in syms]

        sev_vals = [SEVERITY_ENCODING.get(s, 0.5) for s in sevs if s]
        max_sev = max(sev_vals) if sev_vals else 0.5
        severe_n = sum(1 for s in sevs if s == "severe")

        days_list = [v for d in durs if d for v in [_days(d)] if v is not None]
        max_days = max(days_list) if days_list else None
        acute = 1 if _acute(durs) else 0
        chronic = 1 if max_days and max_days > 14 else 0

        red_n = sum(1 for n in names_set if n in RED_FLAG_SYMPTOMS)
        has_red = 1 if red_n > 0 else 0

        syndromes = {f"syndrome_{k}": 1 if v & names_set else 0 for k, v in SYNDROME_DEFINITIONS.items()}

        age = _age(dem.get("age"))
        gender = str(dem.get("gender", "")).lower() or "unknown"
        c = sum(1 for w in ["diabetes", "hypertension", "heart", "lung"] if _has(dem, hist, w))

        return {
            "symptom_codes": codes,
            "symptom_vector": list(set(codes)),
            "symptom_text": text,
            "demographics": dem,
            "history": hist,
            "severities": sevs,
            "durations": durs,
            "symptom_count": len(syms),
            "max_severity": max_sev,
            "has_severe": 1 if severe_n > 0 else 0,
            "severe_count": severe_n,
            "severity_weighted_sum": sum(sev_vals),
            "max_duration_days": max_days,
            "acute_flag": acute,
            "chronic_flag": chronic,
            "has_red_flag": has_red,
            "red_flag_count": red_n,
            "red_flag_severe": 1 if (has_red and severe_n > 0) else 0,
            **syndromes,
            "age": age,
            "age_group": _age_group(age),
            "gender": gender,
            "has_diabetes": 1 if _has(dem, hist, "diabetes") else 0,
            "has_hypertension": 1 if _has(dem, hist, "hypertension") else 0,
            "has_heart_disease": 1 if _has(dem, hist, "heart") else 0,
            "has_lung_disease": 1 if _has(dem, hist, "lung") else 0,
            "comorbidity_count": c,
        }
