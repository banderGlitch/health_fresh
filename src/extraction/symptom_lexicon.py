"""
Symptom lexicon for Phase 1 extraction.

Canonical symptom → list of variations (used by MLNERExtractor for normalization).
VARIATION_TO_CANONICAL: reverse lookup for get_canonical().
"""

# Canonical symptom -> variations (lowercase)
SYMPTOM_LEXICON: dict[str, list[str]] = {
    "fever": ["fever", "feverish", "pyrexia", "high temperature", "temp", "running a temperature"],
    "headache": ["headache", "head pain", "head hurts", "migraine", "cephalgia"],
    "chest pain": ["chest pain", "chest discomfort", "chest tightness", "chest pressure", "burning chest", "chest ache"],
    "cough": ["cough", "coughing", "dry cough", "wet cough", "hacking cough"],
    "shortness of breath": ["shortness of breath", "breathlessness", "dyspnea", "difficulty breathing", "can't breathe", "trouble breathing"],
    "nausea": ["nausea", "nauseous", "nauseated", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "vomit", "throwing up", "emesis", "puking"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "feeling tired", "weakness", "lethargy"],
    "sore throat": ["sore throat", "throat pain", "throat hurts", "pharyngitis"],
    "runny nose": ["runny nose", "rhinorrhea", "stuffy nose", "nasal congestion", "blocked nose"],
    "body ache": ["body ache", "body pain", "muscle ache", "muscle pain", "myalgia", "aches"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "stomachache", "stomach ache", "stomach discomfort"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stools", "watery stools"],
    "dizziness": ["dizziness", "dizzy", "vertigo", "lightheaded", "light-headed"],
    "sweating": ["sweating", "sweats", "perspiration", "night sweats"],
    "chills": ["chills", "shivering", "shivers", "feeling cold"],
    "loss of appetite": ["loss of appetite", "no appetite", "not hungry", "reduced appetite"],
    "insomnia": ["insomnia", "sleeplessness", "can't sleep", "difficulty sleeping"],
    "rash": ["rash", "skin rash", "eruption"],
    "swelling": ["swelling", "swollen", "edema", "oedema"],
    "joint pain": ["joint pain", "joint ache", "arthralgia"],
    "back pain": ["back pain", "backache", "lower back pain"],
    "ear pain": ["ear pain", "earache", "ear ache"],
    "eye pain": ["eye pain", "eye discomfort", "sore eyes"],
    "blurred vision": ["blurred vision", "blurry vision", "vision problems"],
    "palpitations": ["palpitations", "heart racing", "fast heartbeat", "irregular heartbeat"],
    "weight loss": ["weight loss", "losing weight", "unintended weight loss"],
    "coughing blood": ["coughing blood", "blood in cough", "hemoptysis"],
    "blood in stool": ["blood in stool", "bloody stool", "rectal bleeding"],
}

# Build reverse lookup: variation -> canonical
VARIATION_TO_CANONICAL: dict[str, str] = {}
for canonical, variations in SYMPTOM_LEXICON.items():
    for v in variations:
        VARIATION_TO_CANONICAL[v.lower()] = canonical


def get_canonical(symptom_text: str) -> str | None:
    """Return canonical symptom name if found in lexicon."""
    return VARIATION_TO_CANONICAL.get(symptom_text.lower().strip())
