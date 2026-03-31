"""Load and normalize SYNAPSE dataset for RAG indexing."""

from typing import Any

import pandas as pd

from config import MAX_SAMPLES, SYNAPSE_DATA_PATH


def _to_text(row: pd.Series) -> str:
    # This function takes one row of dataset 
    # and converts it into one text string
    # by concatenating the symptoms, gender, age, duration, and severity.
   
    """Build a single searchable text from symptoms + demographics."""
    parts = [str(row.get("Symptoms", "")).strip()]
    parts.append(str(row.get("Gender", "")).strip())
    parts.append(str(row.get("Age", "")).strip())
    parts.append(str(row.get("Duration", "")).strip())
    parts.append(str(row.get("Severity", "")).strip())
    # "Headache, fever | Male | 25 | 2 days | Mild"
    # This text is used to search for embedding text.
    return " | ".join(p for p in parts if p)


def load_synapse_cases(max_samples: int | None = None) -> list[dict[str, Any]]:
    """
    Load SYNAPSE CSV, normalize, and return list of case dicts.
    Each case has: text, symptoms, gender, age, duration, severity, recommendation.
    """
    path = SYNAPSE_DATA_PATH
    if not path or not path.strip():
        raise FileNotFoundError("SYNAPSE_DATA_PATH not set. Point to the SYNAPSE CSV.")
    df = pd.read_csv(path)
    if df.empty:
        return []

    limit = max_samples or MAX_SAMPLES
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    cases = []
    for _, row in df.iterrows():
        rec = str(row.get("Final Recommendation", "")).strip()
        if rec not in ("OTC Drug", "Doctor Consultation"):
            continue
        text = _to_text(row)
        if not text or text == " | | | | ":
            continue
        cases.append({
            "text": text,
            "symptoms": str(row.get("Symptoms", "")).strip(),
            "gender": str(row.get("Gender", "")).strip(),
            "age": str(row.get("Age", "")).strip(),
            "duration": str(row.get("Duration", "")).strip(),
            "severity": str(row.get("Severity", "")).strip(),
            "recommendation": rec,
        })
    return cases
