"""Quick test for RAG triage Phase 1."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.triage import RAGTriagePredictor


def main():
    predictor = RAGTriagePredictor()
    if not predictor.is_available:
        print("Index not built. Run: python scripts/build_index.py")
        return 1

    # Test 1: OTC-like case
    r1 = predictor.predict(
        symptom_text="fever, headache, itching",
        age=25,
        gender="male",
        duration_days=2,
        severity="mild",
    )
    print("Test 1 (fever, headache, mild):", r1["triage_recommendation"], "| confidence:", r1["Confidence"])
    print("  Similar:", [c["recommendation"] for c in r1["similar_cases"]])

    # Test 2: Doctor-like case
    r2 = predictor.predict(
        symptom_text="chest pain, shortness of breath",
        age=55,
        gender="female",
        duration_days=5,
        severity="moderate",
    )
    print("Test 2 (chest pain, breath):", r2["triage_recommendation"], "| confidence:", r2["Confidence"])

    # Test 3: Pipeline features format
    r3 = predictor.predict(features={
        "symptom_text": "fever, cough, sore throat",
        "age": 40,
        "gender": "male",
        "max_duration_days": 1,
        "acute_flag": 1,
        "severities": ["moderate"],
    })
    print("Test 3 (features dict):", r3["triage_recommendation"])
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
