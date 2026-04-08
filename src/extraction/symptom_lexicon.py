"""
Symptom lexicon for Phase 1 extraction.

Canonical symptom → list of variations (used by MLNERExtractor for normalization).
VARIATION_TO_CANONICAL: reverse lookup for get_canonical().

Hand-curated synonyms live in _BASE_SYMPTOM_LEXICON.

Extra wording from the SYNAPSE dataset is matched at runtime via ``symptom_rag.rag_spans``
(overlap on CSV phrases) — no giant static merge file required.
"""

from __future__ import annotations

# Hand-curated: high-signal synonyms and normalization.
_BASE_SYMPTOM_LEXICON: dict[str, list[str]] = {
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

SYMPTOM_LEXICON: dict[str, list[str]] = {k: list(v) for k, v in _BASE_SYMPTOM_LEXICON.items()}

# Build reverse lookup: variation -> canonical
VARIATION_TO_CANONICAL: dict[str, str] = {}
for canonical, variations in SYMPTOM_LEXICON.items():
    for v in variations:
        VARIATION_TO_CANONICAL[v.lower()] = canonical
# Ensure canonical keys resolve to themselves
for canonical in SYMPTOM_LEXICON:
    VARIATION_TO_CANONICAL.setdefault(canonical.lower(), canonical)


def get_canonical(symptom_text: str) -> str | None:
    """Return canonical symptom name if found in lexicon."""
    return VARIATION_TO_CANONICAL.get(symptom_text.lower().strip())


def symptom_mentioned_in_patient_text(canonical_name: str, patient_text: str) -> bool:
    """
    True if some lexicon phrase for this symptom appears in patient-authored text.
    Used to drop NER/lexicon hits that only appear in LLM-merged narrative (hallucinated symptoms).
    RAG-only symptom names (long CSV phrases) fall back to substring match on canonical_name.
    """
    if not canonical_name or not patient_text or not patient_text.strip():
        return False
    text_lower = patient_text.lower()
    variations = SYMPTOM_LEXICON.get(canonical_name)
    if variations is None:
        pl = canonical_name.lower().strip()
        return len(pl) >= 2 and pl in text_lower
    for phrase in [canonical_name] + list(variations):
        pl = phrase.lower().strip()
        if len(pl) < 2:
            continue
        if pl in text_lower:
            return True
    return False


def ground_extraction_dict(extraction_dict: dict, patient_text: str) -> dict:
    """
    Keep only symptoms and negations whose wording appears in patient_text
    (original complaint + follow-up answers, not scribe-merged narrative).
    """
    if not patient_text or not str(patient_text).strip():
        return extraction_dict
    out = dict(extraction_dict)
    symptoms = out.get("symptoms") or []
    out["symptoms"] = [
        s
        for s in symptoms
        if isinstance(s, dict) and symptom_mentioned_in_patient_text(str(s.get("name", "")), patient_text)
    ]
    negated = out.get("negated") or []
    out["negated"] = [
        n
        for n in negated
        if isinstance(n, str) and symptom_mentioned_in_patient_text(n, patient_text)
    ]
    return out
