"""
Test script for SYNAPSE-trained telehealth model.
Validates prediction correctness on the SYNAPSE dataset.

If the saved model (telehealth_model.pkl) fails to load (version mismatch),
trains a fresh model with the same approach and reports metrics.
"""

import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNAPSE_DIR = PROJECT_ROOT / "models" / "synapse_triage"
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv"
VECTORIZER_PATH = SYNAPSE_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = SYNAPSE_DIR / "label_encoder.pkl"
MODEL_PATH = SYNAPSE_DIR / "telehealth_model.pkl"


def load_data():
    """Load SYNAPSE dataset."""
    df = pd.read_csv(CSV_PATH)
    # Drop rows with missing target
    df = df.dropna(subset=["Final Recommendation"])
    df = df[df["Final Recommendation"].isin(["OTC Drug", "Doctor Consultation"])]
    return df


def build_input_text(row):
    """
    Build combined text for vectorizer (matches typical SYNAPSE model training).
    Combines: Symptoms + Gender + Age + Duration + Severity
    """
    parts = [
        str(row.get("Symptoms", "")),
        str(row.get("Gender", "")),
        str(row.get("Age", "")),
        str(row.get("Duration", "")),
        str(row.get("Severity", "")),
    ]
    return " ".join(parts).lower()


def load_or_train_model(df_train, df_test, X_train, X_test, y_train, y_test):
    """Try loading saved model; if fails, train a fresh one."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    # Load vectorizer and label encoder (these usually load fine)
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception as e:
        print(f"[WARN] Could not load vectorizer/label_encoder: {e}")
        vectorizer = None
        label_encoder = None

    model = None
    model_source = "trained_fresh"

    # Try loading saved model
    try:
        model = joblib.load(MODEL_PATH)
        model_source = "saved_telehealth_model.pkl"
    except Exception as e:
        print(f"\n[INFO] Could not load saved model ({e})")
        print("       Training a fresh model with same approach (TfidfVectorizer + LogisticRegression)...\n")

    # If model failed to load, train fresh
    if model is None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        label_encoder = LabelEncoder()

        X_train_vec = vectorizer.fit_transform(df_train["input_text"])
        X_test_vec = vectorizer.transform(df_test["input_text"])
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train_enc)

        # Override for evaluation
        X_train = X_train_vec
        X_test = X_test_vec
        y_train = y_train_enc
        y_test = y_test_enc
    else:
        # Use loaded vectorizer
        X_train = vectorizer.transform(df_train["input_text"])
        X_test = vectorizer.transform(df_test["input_text"])
        y_train_enc = label_encoder.transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
        y_train = y_train_enc
        y_test = y_test_enc

    return model, vectorizer, label_encoder, X_train, X_test, y_train, y_test, model_source


def run_evaluation():
    """Run full evaluation."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("=" * 60)
    print("SYNAPSE Model Test")
    print("=" * 60)

    # Load data
    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"    Rows: {len(df)}")
    print(f"    Target distribution:\n{df['Final Recommendation'].value_counts()}")

    # Build input text
    df["input_text"] = df.apply(build_input_text, axis=1)
    X = df["input_text"]
    y = df["Final Recommendation"]

    # Split 80/20
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    X_train_text = df_train["input_text"]
    X_test_text = df_test["input_text"]
    y_train = df_train["Final Recommendation"]
    y_test = df_test["Final Recommendation"]

    # Load or train model
    print("\n[2] Loading/training model...")
    result = load_or_train_model(
        df_train, df_test,
        X_train_text, X_test_text,
        y_train, y_test
    )
    model, vectorizer, label_encoder, X_train, X_test, y_train, y_test, model_source = result

    print(f"    Model source: {model_source}")

    # Predict
    print("\n[3] Running predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Use string labels for evaluation (y_test may be encoded)
    y_test_labels = df_test["Final Recommendation"].values

    # Metrics
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))
    print("(Rows=Actual, Cols=Predicted)")

    # Sample predictions
    print("\n" + "-" * 60)
    print("Sample Predictions (first 10 test cases)")
    print("-" * 60)
    for i in range(min(10, len(df_test))):
        row = df_test.iloc[i]
        actual = row["Final Recommendation"]
        pred = y_pred_labels[i]
        correct = "OK" if actual == pred else "MISMATCH"
        print(f"\n{i+1}. Symptoms: {str(row['Symptoms'])[:80]}...")
        print(f"   Severity: {row['Severity']} | Actual: {actual} | Predicted: {pred} | {correct}")

    # Critical safety check: severe + red-flag symptoms should predict Doctor
    print("\n" + "-" * 60)
    print("Safety Check: Severe cases with red-flag symptoms")
    print("-" * 60)
    red_flags = ["chest pain", "shortness breath", "blood", "unconscious", "severe"]
    severe_doctor = df_test[
        (df_test["Severity"] == "Severe") & 
        (df_test["input_text"].str.lower().str.contains("|".join(red_flags), regex=True, na=False))
    ]
    if len(severe_doctor) > 0:
        severe_preds = []
        for _, row in severe_doctor.iterrows():
            vec = vectorizer.transform([row["input_text"]])
            pred_enc = model.predict(vec)[0]
            pred_label = label_encoder.inverse_transform([pred_enc])[0]
            severe_preds.append(pred_label)
        correct_doctor = sum(1 for p in severe_preds if p == "Doctor Consultation")
        print(f"Found {len(severe_preds)} severe+red-flag cases. Correctly predicted Doctor: {correct_doctor}/{len(severe_preds)}")
        if correct_doctor < len(severe_preds):
            print("WARNING: Some high-risk cases predicted as OTC - safety concern!")
    else:
        print("No severe+red-flag cases in test subset.")

    return accuracy


if __name__ == "__main__":
    run_evaluation()
