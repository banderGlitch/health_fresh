"""
Test finetuned triage models: fine-tuned SYNAPSE triage risk model.

label_encoders.pkl + risk_model.pkl = alternative to models/synapse_triage/
- Input: 5 features (Symptoms category, Gender, Age, Duration, Severity)
- Output: OTC Drug | Doctor Consultation | Emergency (3-way, vs SYNAPSE's 2-way)

Standalone test only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models" / "finetuned_triage"


def test_finetuned_triage():
    """Test that models load and inspect their structure."""
    import joblib

    print("=" * 50)
    print("Testing models in models/finetuned_triage/")
    print("=" * 50)

    le_path = MODELS_DIR / "label_encoders.pkl"
    rm_path = MODELS_DIR / "risk_model.pkl"
    if not le_path.exists():
        print(f"ERROR: {le_path} not found")
        return
    if not rm_path.exists():
        print(f"ERROR: {rm_path} not found")
        return
    print(f"Found: {le_path.name} ({le_path.stat().st_size} bytes)")
    print(f"Found: {rm_path.name} ({rm_path.stat().st_size} bytes)")
    print()

    print("[1] Loading label_encoders.pkl ...")
    try:
        le = joblib.load(le_path)
        print(f"    Type: {type(le).__name__}")
        if isinstance(le, dict):
            for k, v in le.items():
                print(f"    Key '{k}': {type(v).__name__}")
                if hasattr(v, "classes_"):
                    print(f"      classes: {list(v.classes_)[:10]}...")
        elif hasattr(le, "classes_"):
            print(f"    Classes: {list(le.classes_)[:15]}")
        print("    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        return
    print()

    print("[2] Loading risk_model.pkl ...")
    try:
        rm = joblib.load(rm_path)
        print(f"    Type: {type(rm).__name__}")
        if hasattr(rm, "predict"):
            print("    Has predict()")
        if hasattr(rm, "n_features_in_"):
            print(f"    n_features_in: {rm.n_features_in_}")
        if hasattr(rm, "classes_"):
            print(f"    classes: {rm.classes_}")
        print("    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        return
    print()

    symptom_classes = list(le["Symptoms"].classes_)
    print(f"[3] Symptom categories ({len(symptom_classes)}):")
    for i, s in enumerate(symptom_classes):
        print(f"      {i}: {s}")
    print()

    print("[4] Test predictions (finetuned triage model) ...")
    test_cases = [
        ("Fever and mild headache", "Male", "31-50", "3-5 days", "Mild"),
        ("Chest tightness and sweating", "Female", "50+", "1-2 days", "Severe"),
        ("Cough and cold", "Male", "16-30", "1 week", "Mild"),
        ("High fever for 5 days", "Female", "6-15", "3-5 days", "Moderate"),
        ("Breathing shortness", "Male", "50+", "1-2 days", "Severe"),
    ]
    try:
        import numpy as np
        for symptom, gender, age, duration, severity in test_cases:
            symptom_enc = le["Symptoms"].transform([symptom])[0]
            gender_enc = le["Gender"].transform([gender])[0]
            age_enc = le["Age"].transform([age])[0]
            duration_enc = le["Duration"].transform([duration])[0]
            severity_enc = le["Severity"].transform([severity])[0]
            X = np.array([[symptom_enc, gender_enc, age_enc, duration_enc, severity_enc]])
            pred = rm.predict(X)[0]
            label = le["Final Recommendation"].inverse_transform([pred])[0]
            print(f"    {symptom[:30]:30} | {gender:6} | {age:6} | {duration:10} | {severity:8} -> {label}")
        print("    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
    print()
    print("Finetuned triage model: TEST COMPLETE.")


if __name__ == "__main__":
    test_finetuned_triage()
