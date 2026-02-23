"""
Phase 1 extraction tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction import NERExtractor


def test_basic_symptoms():
    extractor = NERExtractor()
    result = extractor.extract("I have fever and headache.")
    assert len(result.symptoms) == 2
    names = {s.name for s in result.symptoms}
    assert "fever" in names
    assert "headache" in names


def test_duration():
    extractor = NERExtractor()
    result = extractor.extract("I have had fever for 3 days.")
    assert len(result.symptoms) >= 1
    fever = next(s for s in result.symptoms if s.name == "fever")
    assert fever.duration == "3 days"


def test_severity():
    extractor = NERExtractor()
    result = extractor.extract("I have mild headache and severe chest pain.")
    assert len(result.symptoms) >= 2
    for s in result.symptoms:
        if s.name == "headache":
            assert s.severity == "mild"
        if s.name == "chest pain":
            assert s.severity == "severe"


def test_negation():
    extractor = NERExtractor()
    result = extractor.extract("I have fever for 3 days. No vomiting or shortness of breath.")
    assert "vomiting" in result.negated
    assert "shortness of breath" in result.negated
    assert len(result.symptoms) >= 1
    names = {s.name for s in result.symptoms}
    assert "vomiting" not in names
    assert "shortness of breath" not in names


def test_associated_factors():
    extractor = NERExtractor()
    result = extractor.extract("I have fever with nausea and sweating.")
    assert len(result.symptoms) >= 1
    fever = next(s for s in result.symptoms if s.name == "fever")
    assert "nausea" in fever.associated_factors or "sweating" in fever.associated_factors


def test_full_example():
    extractor = NERExtractor()
    text = "I've had fever for 3 days and a mild headache. No vomiting."
    result = extractor.extract(text)
    d = extractor.to_dict(result)

    assert len(d["symptoms"]) >= 2
    assert "vomiting" in d["negated"]
    fever = next((s for s in d["symptoms"] if s["name"] == "fever"), None)
    if fever:
        assert fever["duration"] == "3 days"
    headache = next((s for s in d["symptoms"] if s["name"] == "headache"), None)
    if headache:
        assert headache["severity"] == "mild"


if __name__ == "__main__":
    test_basic_symptoms()
    print("OK basic_symptoms")
    test_duration()
    print("OK duration")
    test_severity()
    print("OK severity")
    test_negation()
    print("OK negation")
    test_associated_factors()
    print("OK associated_factors")
    test_full_example()
    print("OK full_example")
    print("\nAll Phase 1 extraction tests passed.")
