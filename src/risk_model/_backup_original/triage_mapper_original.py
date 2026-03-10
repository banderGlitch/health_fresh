"""
BACKUP - Original triage_mapper.py
Restore: copy to ../triage_mapper.py
"""

from typing import Any

TRIAGE_SYMPTOMS = [
    "Breathing shortness", "Chest tightness and sweating", "Cold and throat irritation",
    "Cough and cold", "Fever and mild headache", "Headache since one week",
    "High blood pressure symptoms", "High fever for 5 days", "Low grade fever",
    "Mild back pain", "Mild fever", "Mild skin rash", "Persistent cough",
    "Severe abdominal pain", "Severe allergic reaction", "Severe breathing difficulty",
    "Severe chest pain", "Severe migraine", "Stomach pain and vomiting",
    "Vomiting and dehydration",
]

_SYMPTOM_MAPPING_RULES = [
    ({"chest pain", "chest tightness"}, "severe", "Severe chest pain"),
    ({"chest tightness", "sweating", "chest pain"}, None, "Chest tightness and sweating"),
    ({"shortness of breath", "breathing difficulty", "dyspnea"}, "severe", "Severe breathing difficulty"),
    ({"shortness of breath", "breath", "breathing"}, None, "Breathing shortness"),
    ({"abdominal pain", "stomach pain"}, "severe", "Severe abdominal pain"),
    ({"stomach pain", "vomiting", "abdominal"}, None, "Stomach pain and vomiting"),
    ({"vomiting", "dehydration", "diarrhea"}, None, "Vomiting and dehydration"),
    ({"allergic", "allergy", "hives", "swelling", "anaphylaxis"}, None, "Severe allergic reaction"),
    ({"migraine", "severe headache"}, "severe", "Severe migraine"),
    ({"headache"}, "severe", "Severe migraine"),
    ({"high fever", "fever 5 days", "fever for 5 days"}, None, "High fever for 5 days"),
    ({"fever", "headache"}, None, "Fever and mild headache"),
    ({"headache", "week"}, None, "Headache since one week"),
    ({"cough", "cold", "runny nose", "congestion"}, None, "Cough and cold"),
    ({"sore throat", "throat", "cold", "irritation"}, None, "Cold and throat irritation"),
    ({"cough", "persistent", "chronic cough"}, None, "Persistent cough"),
    ({"cough", "cold"}, None, "Cough and cold"),
    ({"low grade fever", "mild fever", "slight fever"}, None, "Low grade fever"),
    ({"mild fever", "fever"}, None, "Mild fever"),
    ({"fever"}, None, "Mild fever"),
    ({"back pain", "backache"}, None, "Mild back pain"),
    ({"skin rash", "rash", "itching"}, None, "Mild skin rash"),
    ({"blood pressure", "hypertension", "high bp"}, None, "High blood pressure symptoms"),
]

_DEFAULT_SYMPTOM = "Cough and cold"


def _normalize(s: str) -> str:
    return s.lower().strip()


def map_symptoms_to_category(symptom_names: list[str], severities: list[str | None] | None = None, max_duration_days: float | None = None) -> str:
    severities = severities or []
    has_severe = any(s and _normalize(str(s)) == "severe" for s in severities)
    symptom_set = {_normalize(n) for n in symptom_names if n}
    all_tokens = set()
    for n in symptom_names:
        if n:
            for t in _normalize(n).split():
                if len(t) > 2:
                    all_tokens.add(t)
    combined = symptom_set | all_tokens

    for keywords, severity_hint, category in _SYMPTOM_MAPPING_RULES:
        if not keywords.intersection(combined):
            continue
        if severity_hint == "severe" and not has_severe:
            continue
        if severity_hint == "mild" and has_severe:
            continue
        if category == "High fever for 5 days" and max_duration_days is not None:
            if max_duration_days < 4 or max_duration_days > 7:
                continue
        return category

    return _DEFAULT_SYMPTOM


def map_age_to_triage(age: int) -> str:
    if age < 6:
        return "6-15"
    if age <= 15:
        return "6-15"
    if age <= 30:
        return "16-30"
    if age <= 50:
        return "31-50"
    return "50+"


def map_duration_to_triage(max_days: float | None, acute_flag: int) -> str:
    if acute_flag or (max_days is not None and max_days <= 2):
        return "1-2 days"
    if max_days is not None and max_days <= 5:
        return "3-5 days"
    if max_days is not None and max_days <= 10:
        return "1 week"
    return "2+ weeks"


def map_severity_to_triage(severities: list[str | None]) -> str:
    mapping = {"mild": 0, "moderate": 1, "severe": 2}
    max_val = 0
    for s in (severities or []):
        if s:
            max_val = max(max_val, mapping.get(str(s).lower(), 1))
    return ["Mild", "Moderate", "Severe"][max_val]


def map_gender_to_triage(gender: str) -> str:
    g = str(gender).lower().strip()
    if g in ("male", "m"):
        return "Male"
    return "Female"


def features_to_triage_input(features: dict[str, Any]) -> dict[str, Any] | None:
    symptom_text = features.get("symptom_text") or ""
    symptom_names = [s.strip() for s in symptom_text.split(",") if s.strip()]
    if not symptom_names:
        symptom_names = ["general discomfort"]

    severities = features.get("severities") or []
    max_days = features.get("max_duration_days")
    acute_flag = features.get("acute_flag", 0)

    symptom_category = map_symptoms_to_category(symptom_names, severities, max_days)
    age = features.get("age", 40)
    gender = map_gender_to_triage(features.get("gender", "unknown"))
    age_bucket = map_age_to_triage(age)
    duration = map_duration_to_triage(max_days, acute_flag)
    severity = map_severity_to_triage(severities)

    return {
        "symptom_category": symptom_category,
        "gender": gender,
        "age_bucket": age_bucket,
        "duration": duration,
        "severity": severity,
    }
