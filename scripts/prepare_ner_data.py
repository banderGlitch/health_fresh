"""
Prepare NER training data from symptom lexicon.
Generates synthetic sentences with BIO labels (B-SYMPTOM, I-SYMPTOM, O).
Output: CoNLL format (token TAB label per line, blank line between sentences).
"""

import random
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.symptom_lexicon import SYMPTOM_LEXICON


# Sentence templates: {symptom} is placeholder for symptom phrase
TEMPLATES = [
    "I have {symptom}.",
    "I have {symptom} and feel unwell.",
    "I've had {symptom} for a few days.",
    "Patient presents with {symptom}.",
    "Main complaint is {symptom}.",
    "Experiencing {symptom} since yesterday.",
    "I have mild {symptom}.",
    "I have severe {symptom}.",
    "I have moderate {symptom}.",
    "I have {symptom} with nausea.",
    "I have {symptom} and headache.",
    "I have {symptom} for 3 days.",
    "I have {symptom} and fatigue.",
    "I've been having {symptom}.",
    "I have {symptom} and body ache.",
    "I have {symptom} and cough.",
    "I have {symptom} and sore throat.",
    "I have {symptom} and runny nose.",
    "I have {symptom} and chills.",
    "I have {symptom} and sweating.",
    "I have {symptom} and dizziness.",
    "I have {symptom} and loss of appetite.",
    "I have {symptom} and vomiting.",
    "I have {symptom} and diarrhea.",
    "I have {symptom} and abdominal pain.",
    "I have {symptom} and chest pain.",
    "I have {symptom} and shortness of breath.",
    "I have {symptom} and joint pain.",
    "I have {symptom} and back pain.",
    "I have {symptom} and rash.",
    "I have {symptom} and swelling.",
    "I have {symptom} and blurred vision.",
    "I have {symptom} and palpitations.",
    "I have {symptom} and weight loss.",
    "I have {symptom} and insomnia.",
    "I have {symptom} and ear pain.",
    "I have {symptom} and eye pain.",
    "I have {symptom} and coughing blood.",
    "I have {symptom} and blood in stool.",
    "I have {symptom} for 3 days.",
    "I have {symptom} for a week.",
    "I have {symptom} since yesterday.",
    "I have {symptom} for 2 weeks.",
    "I have {symptom} for 5 days.",
    "I have {symptom} and fatigue for 3 days.",
    "I have {symptom} and headache for a week.",
    "I have mild {symptom} for 3 days.",
    "I have severe {symptom} for a week.",
    "I have moderate {symptom}.",
    "No {symptom}.",
    "I have fever but no {symptom}.",
    "I have headache but no {symptom}.",
    "I have fever and headache. No {symptom}.",
    "I have {symptom} and {symptom2}.",
]

# Multi-symptom templates (two symptoms)
MULTI_TEMPLATES = [
    "I have {s1} and {s2}.",
    "I have {s1} with {s2}.",
    "I have {s1} and {s2} for 3 days.",
    "I have {s1} and {s2} and feel tired.",
    "I have {s1} and {s2} since yesterday.",
    "I have mild {s1} and severe {s2}.",
    "I have {s1} and {s2} and nausea.",
    "I have {s1} and {s2} and fatigue.",
]


def tokenize_with_labels(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    """
    Tokenize text by whitespace and assign BIO labels.
    spans: list of (start, end, canonical_name) for symptom spans.
    """
    tokens = text.split()
    result: list[tuple[str, str]] = []
    pos = 0

    for token in tokens:
        # Find token boundaries in original text
        start = text.find(token, pos)
        if start == -1:
            start = pos
        end = start + len(token)
        pos = end

        # Check if this token overlaps any symptom span
        label = "O"
        for s_start, s_end, _ in spans:
            if start < s_end and end > s_start:
                # Overlap
                if start == s_start:
                    label = "B-SYMPTOM"
                else:
                    label = "I-SYMPTOM"
                break

        result.append((token, label))

    return result


def generate_single_symptom_examples() -> list[list[tuple[str, str]]]:
    """Generate examples with one symptom per sentence."""
    examples = []
    for canonical, variations in SYMPTOM_LEXICON.items():
        for variant in variations:
            for template in TEMPLATES:
                if "{symptom}" in template and "{symptom2}" not in template:
                    # Skip negation templates for this symptom
                    if "No {symptom}" in template or "no {symptom}" in template:
                        continue
                    sent = template.replace("{symptom}", variant)
                    # Find span of variant in sentence
                    idx = sent.lower().find(variant.lower())
                    if idx >= 0:
                        spans = [(idx, idx + len(variant), canonical)]
                        labeled = tokenize_with_labels(sent, spans)
                        if any(l == "B-SYMPTOM" or l == "I-SYMPTOM" for _, l in labeled):
                            examples.append(labeled)
    return examples


def generate_multi_symptom_examples() -> list[list[tuple[str, str]]]:
    """Generate examples with two symptoms per sentence."""
    examples = []
    canonicals = list(SYMPTOM_LEXICON.keys())
    for _ in range(500):  # Limit to avoid huge dataset
        s1, s2 = random.sample(canonicals, 2)
        if s1 == s2:
            continue
        v1 = random.choice(SYMPTOM_LEXICON[s1])
        v2 = random.choice(SYMPTOM_LEXICON[s2])
        if v1 == v2:
            continue
        template = random.choice(MULTI_TEMPLATES)
        sent = template.replace("{s1}", v1).replace("{s2}", v2)
        if "{symptom}" in sent or "{symptom2}" in sent:
            continue
        # Find spans
        idx1 = sent.lower().find(v1.lower())
        idx2 = sent.lower().find(v2.lower())
        if idx1 >= 0 and idx2 >= 0 and idx1 != idx2:
            spans = [(idx1, idx1 + len(v1), s1), (idx2, idx2 + len(v2), s2)]
            spans.sort(key=lambda x: x[0])
            labeled = tokenize_with_labels(sent, spans)
            if sum(1 for _, l in labeled if l != "O") >= 2:
                examples.append(labeled)
    return examples


def generate_negation_examples() -> list[list[tuple[str, str]]]:
    """Generate examples with negated symptoms (negated symptom gets O)."""
    examples = []
    for canonical, variations in SYMPTOM_LEXICON.items():
        for variant in variations[:3]:  # Limit
            # "No X" - nothing to label (all O)
            sent = f"No {variant}."
            spans = []
            examples.append(tokenize_with_labels(sent, spans))
            # "I have fever but no X" - label "fever" only
            sent = f"I have fever but no {variant}."
            idx = sent.lower().find("fever")
            if idx >= 0:
                spans = [(idx, idx + 5, "fever")]
                examples.append(tokenize_with_labels(sent, spans))
    return examples


def to_conll(examples: list[list[tuple[str, str]]], path: Path) -> None:
    """Write examples to CoNLL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in examples:
            for token, label in sent:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "ner"
    out_dir.mkdir(parents=True, exist_ok=True)

    single = generate_single_symptom_examples()
    multi = generate_multi_symptom_examples()
    neg = generate_negation_examples()

    all_examples = single + multi + neg
    random.shuffle(all_examples)

    # Split 80/10/10
    n = len(all_examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = all_examples[:train_end]
    val_data = all_examples[train_end:val_end]
    test_data = all_examples[val_end:]

    to_conll(train_data, out_dir / "train.txt")
    to_conll(val_data, out_dir / "val.txt")
    to_conll(test_data, out_dir / "test.txt")

    labels_file = out_dir / "labels.txt"
    labels_file.write_text("O\nB-SYMPTOM\nI-SYMPTOM\n")

    print(f"Generated NER training data in {out_dir}")
    print(f"  Train: {len(train_data)} sentences")
    print(f"  Val:   {len(val_data)} sentences")
    print(f"  Test:  {len(test_data)} sentences")
    print(f"  Labels: O, B-SYMPTOM, I-SYMPTOM")


if __name__ == "__main__":
    main()
