"""Test RAG predictor with real SYNAPSE dataset cases."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.triage import RAGTriagePredictor

# Map SYNAPSE age/duration to predictor format
def _age_to_num(age_str):
    if "below 5" in age_str: return 3
    if "6-15" in age_str: return 10
    if "16-45" in age_str or "16-60" in age_str: return 35
    if "above 45" in age_str or "above 60" in age_str: return 65
    return 35

def _duration_to_days(dur_str):
    if "Less than" in dur_str: return 2
    return 5

def _acute(dur_str):
    return "Less than" in dur_str

cases = [
    # OTC Drug (expected)
    ("flu like illness | Female | above 45 years | Greater than 3 days | Mild", "OTC Drug"),
    ("Constipation | Female | below 5 years | Greater than 3 days | Moderate", "OTC Drug"),
    ("Skin pain | Female | above 45 years | Greater than 3 days | Moderate", "OTC Drug"),
    ("Abdominal pain, Feeling sick, vomiting | Female | below 5 years | Greater than 3 days | Mild", "OTC Drug"),
    ("Abdominal pain | Female | 16-45 years | Greater than 3 days | Mild", "OTC Drug"),
    ("Fever, headache, itching | Male | 6-15 years | Less than 3 days | Mild", "OTC Drug"),
    ("Fever, Abdominal pain, Sore throat, Rash | Male | 16-60 years | Greater than 3 days | Mild", "OTC Drug"),
    ("Abdominal pain, Feeling sick | Female | above 45 years | Less than 3 days | Moderate", "OTC Drug"),
    ("Skin color change | Female | 16-45 years | Less than 3 days | Mild", "OTC Drug"),
    ("itchy skin rash, Blotchy | Female | 16-45 years | Less than 3 days | Moderate", "OTC Drug"),
    ("fever, Feeling sick, vomiting | Female | 6-15 years | Less than 3 days | Mild", "OTC Drug"),
    # Doctor Consultation (expected)
    ("Red firm lumps, Scaly patches with irregular borders | Male | above 60 years | Less than 3 days | Mild", "Doctor Consultation"),
    ("Shortness breath, Unwanted weight loss, pain with swallowing, Unexplained ear discomfort, Hoarse voice, lump on neck, Coughing blood | Male | 6-15 years | Less than 3 days | Moderate", "Doctor Consultation"),
    ("blurred vision, sensitivity to light, Seeing rainbows around lights | Female | 16-45 years | Less than 3 days | Moderate", "Doctor Consultation"),
    ("Abdominal pain, vomiting, Pins needles feeling in fingers, Pins needles feeling in toes, Garlic odor in breath, Skin redness, swelling skin, Darkening of skin | Female | above 45 years | Greater than 3 days | Mild", "Doctor Consultation"),
    ("Feeling tired, headache, Unexplained weight change, Fast heart rate, Swelling Leg, Abnormal hair growth, New purplish stretch marks | Female | above 45 years | Greater than 3 days | Moderate", "Doctor Consultation"),
    ("Chest pain, Shortness breath, Fast heart rate, Feeling lightheaded, sensation of spinning | Female | above 45 years | Greater than 3 days | Moderate", "Doctor Consultation"),
    ("Blood urine | Female | above 45 years | Greater than 3 days | Moderate", "Doctor Consultation"),
    ("Shortness breath, Abdominal pain, Chest pain | Male | below 5 years | Less than 3 days | Severe", "Doctor Consultation"),
    ("headache, neck pain | Male | below 5 years | Less than 3 days | Mild", "Doctor Consultation"),
    ("Fever, Shortness breath, Feeling tired, Unwanted weight loss, Feeling unwell | Male | above 60 years | Less than 3 days | Moderate", "Doctor Consultation"),
    ("False beliefs, Seeing, hearing things are not there | Male | 6-15 years | Greater than 3 days | Mild", "Doctor Consultation"),
    ("Fever, coughing, whoop sound after coughing, Runny nose, Sneezing, A pause in breathing | Male | 6-15 years | Less than 3 days | Moderate", "Doctor Consultation"),
]

predictor = RAGTriagePredictor()
if not predictor.is_available:
    print("Index not built. Run: python scripts/build_index.py")
    sys.exit(1)

print("Testing RAG predictor with SYNAPSE dataset cases\n")
print("=" * 80)

for i, (text, expected) in enumerate(cases, 1):
    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 5:
        continue
    symptoms, gender, age_str, dur_str, severity = parts[0], parts[1], parts[2], parts[3], parts[4]
    age = _age_to_num(age_str)
    duration_days = _duration_to_days(dur_str)
    acute = _acute(dur_str)

    r = predictor.predict(
        symptom_text=symptoms,
        age=age,
        gender=gender,
        duration_days=duration_days,
        severity=severity,
        acute=acute,
    )
    got = r["triage_recommendation"]
    match = "OK" if got == expected else "MISMATCH"
    print(f"\nCase {i}: {match}")
    print(f"  Input: {symptoms[:60]}... | {gender} | {age_str} | {dur_str} | {severity}")
    print(f"  Expected: {expected} | Got: {got} | Confidence: {r['Confidence']}")

print("\n" + "=" * 80)
print("Done")
