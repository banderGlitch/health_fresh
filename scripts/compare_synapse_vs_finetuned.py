"""
Compare SYNAPSE triage vs finetuned triage model on equivalent test cases.

SYNAPSE: free-text symptoms (TF-IDF) -> OTC | Doctor (2-way)
Finetuned triage: 20 symptom categories -> OTC | Doctor | Emergency (3-way)

Usage: python scripts/compare_synapse_vs_finetuned.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def run_comparison():
    test_cases = [
        ("Fever and mild headache", "Fever and mild headache", "Male", "31-50", "3-5 days", "Mild"),
        ("Chest tightness and sweating", "Chest tightness and sweating", "Female", "50+", "1-2 days", "Severe"),
        ("Cough and cold", "Cough and cold", "Male", "16-30", "1 week", "Mild"),
        ("High fever for 5 days", "High fever for 5 days", "Female", "6-15", "3-5 days", "Moderate"),
        ("Breathing shortness", "Breathing shortness", "Male", "50+", "1-2 days", "Severe"),
    ]

    print("=" * 80)
    print("SYNAPSE vs finetuned triage comparison")
    print("=" * 80)

    synapse_available = False
    synapse_dir = PROJECT_ROOT / "models" / "synapse_triage"
    import joblib
    import numpy as np
    from scipy.sparse import hstack, csr_matrix

    if (synapse_dir / "telehealth_model.pkl").exists():
        try:
            model = joblib.load(synapse_dir / "telehealth_model.pkl")
            vectorizer = joblib.load(synapse_dir / "vectorizer.pkl")
            label_enc = joblib.load(synapse_dir / "label_encoder.pkl")
            scaler = joblib.load(synapse_dir / "scaler.pkl")
            synapse_available = True
        except Exception as e:
            print(f"SYNAPSE load failed: {e}")
    else:
        print("SYNAPSE model files not found.")

    triage_available = False
    triage_dir = PROJECT_ROOT / "models" / "finetuned_triage"
    if (triage_dir / "risk_model.pkl").exists() and (triage_dir / "label_encoders.pkl").exists():
        try:
            le = joblib.load(triage_dir / "label_encoders.pkl")
            rm = joblib.load(triage_dir / "risk_model.pkl")
            triage_available = True
        except Exception as e:
            print(f"Finetuned triage load failed: {e}")
    else:
        print("Finetuned triage model files not found.")

    if not synapse_available or not triage_available:
        print("\nSkipping comparison - one or both models unavailable.")
        return

    age_to_synapse = {"16-30": 0, "31-50": 0, "50+": 4, "6-15": 2}
    dur_to_synapse = {"1-2 days": 1, "3-5 days": 1, "1 week": 0, "2+ weeks": 0}
    sev_to_synapse = {"Mild": 0, "Moderate": 1, "Severe": 2}

    print("\n{:<35} | {:^12} | {:^14}".format("Case", "SYNAPSE (2-way)", "Finetuned (3-way)"))
    print("-" * 80)

    for symptom_text, symptom_cat, gender, age_bucket, duration, severity in test_cases:
        g_enc = 1 if gender.lower() == "male" else 0
        a_enc = age_to_synapse.get(age_bucket, 0)
        d_enc = dur_to_synapse.get(duration, 0)
        s_enc = sev_to_synapse.get(severity, 1)
        text_feat = vectorizer.transform([symptom_text])
        numeric = np.array([[g_enc, a_enc, d_enc, s_enc]])
        numeric_scaled = scaler.transform(numeric)
        X_s = hstack([text_feat, csr_matrix(numeric_scaled)])
        pred_s = model.predict(X_s)[0]
        out_synapse = label_enc.inverse_transform([pred_s])[0]

        s_enc_s = le["Symptoms"].transform([symptom_cat])[0]
        g_enc_s = le["Gender"].transform([gender])[0]
        a_enc_s = le["Age"].transform([age_bucket])[0]
        d_enc_s = le["Duration"].transform([duration])[0]
        sev_enc_s = le["Severity"].transform([severity])[0]
        X_sh = np.array([[s_enc_s, g_enc_s, a_enc_s, d_enc_s, sev_enc_s]])
        pred_sh = rm.predict(X_sh)[0]
        out_triage = le["Final Recommendation"].inverse_transform([pred_sh])[0]

        short = (symptom_text[:32] + "..") if len(symptom_text) > 32 else symptom_text
        print("{:<35} | {:^12} | {:^14}".format(short, out_synapse, out_triage))

    print("-" * 80)
    print("\nKey difference: finetuned triage adds Emergency for severe acute cases.")


if __name__ == "__main__":
    run_comparison()
