"""
Standalone test for the trained in-house NER model.
Run after: python scripts/train_ner.py

Usage:
  python scripts/test_ml_ner.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

MODEL_DIR = project_root / "models" / "ner_symptom"

# Test sentences
TEST_SENTENCES = [
    "I have fever and headache.",
    "I've had chest pain for 3 days.",
    "Patient presents with severe cough and shortness of breath.",
    "I have mild nausea and vomiting.",
    "I have fatigue and body ache for a week.",
    "I have fever but no cough.",
    "I have sore throat and runny nose.",
    "I have abdominal pain and diarrhea.",
    "I have dizziness and sweating.",
    "I have joint pain and back pain.",
]


def run_test():
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        print("Model not found. Train first: python scripts/train_ner.py")
        sys.exit(1)

    try:
        from transformers import pipeline
    except ImportError:
        print("Install: pip install transformers torch")
        sys.exit(1)

    print("Loading trained model...")
    ner = pipeline(
        "ner",
        model=str(MODEL_DIR),
        aggregation_strategy="simple",  # Merge B-I into spans
    )

    print("\n" + "=" * 60)
    print("Testing in-house NER model")
    print("=" * 60)

    for text in TEST_SENTENCES:
        results = ner(text)
        # Results: list of {word, entity_group, score, start, end}
        symptoms = [r["word"].strip() for r in results] if results else []

        print(f"\nInput:  {text}")
        print(f"Found:  {symptoms}")
        if results:
            for r in results:
                print(f"        -> {r['word']!r} ({r.get('entity_group', r.get('entity', ''))} score={r.get('score', 0):.2f})")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    run_test()
