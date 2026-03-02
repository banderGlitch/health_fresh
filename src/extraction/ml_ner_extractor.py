"""
Phase 1 (ML): In-house trained NER model.

- Symptom spans: DistilBERT token classification (models/ner_symptom/)
- Duration, severity, negation: rule-based post-processing via NERExtractor
- Same ExtractionResult format as NERExtractor for pipeline compatibility
"""

import re
from pathlib import Path

from .ner_extractor import ExtractedSymptom, ExtractionResult
from .ner_extractor import NERExtractor  # Used for duration, severity, negation
from .symptom_lexicon import get_canonical


class MLNERExtractor:
    """
    ML-based NER: Uses in-house trained DistilBERT for symptom spans.
    Post-processes with rule-based duration, severity, negation.
    """

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent.parent / "models" / "ner_symptom"
        self.model_path = Path(model_path)
        self._pipeline = None
        self._rule_extractor = NERExtractor()  # For duration, severity, negation

    @property
    def pipeline(self):
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Run: python scripts/train_ner.py"
                )
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=str(self.model_path),
                aggregation_strategy="simple",
            )
        return self._pipeline

    def _merge_spans_to_symptoms(self, text: str, ner_results: list) -> list[tuple[str, int, int]]:
        """Convert NER spans to (canonical_name, start, end)."""
        found: list[tuple[str, int, int]] = []
        for r in ner_results:
            word = r.get("word", "").strip()
            start = r.get("start", 0)
            end = r.get("end", len(text))
            canonical = get_canonical(word) or word
            found.append((canonical, start, end))
        return found

    def extract(self, text: str) -> ExtractionResult:
        """Extract symptoms using ML model + rule-based post-processing."""
        text_clean = " ".join(text.split())
        if not text_clean.strip():
            return ExtractionResult(symptoms=[], negated=[])

        try:
            ner_results = self.pipeline(text_clean)
        except Exception:
            return ExtractionResult(symptoms=[], negated=[])

        if not ner_results:
            return ExtractionResult(symptoms=[], negated=[])

        found = self._merge_spans_to_symptoms(text_clean, ner_results)

        # Use rule-based for duration, severity, negation
        negated = self._rule_extractor._extract_negations(text_clean)
        negated_set = {n.lower() for n in negated}

        symptoms: list[ExtractedSymptom] = []
        for name, start, end in found:
            if name.lower() in negated_set:
                continue
            left_context = text_clean[max(0, start - 50) : start].lower()
            if re.search(r"\b(no|not|without|denies?)\s+", left_context):
                continue

            duration = self._rule_extractor._extract_duration_near(text_clean, start, end)
            severity = self._rule_extractor._extract_severity_near(text_clean, start, end)
            associated = [
                a for a in self._rule_extractor._get_associated_symptoms(name, found)
                if a.lower() not in negated_set
            ]

            symptoms.append(ExtractedSymptom(
                name=name,
                duration=duration,
                severity=severity,
                associated_factors=associated,
            ))

        return ExtractionResult(symptoms=symptoms, negated=negated)

    def to_dict(self, result: ExtractionResult) -> dict:
        """Convert to pipeline output format."""
        return {
            "symptoms": [
                {
                    "name": s.name,
                    "duration": s.duration,
                    "severity": s.severity,
                    "associated_factors": s.associated_factors,
                }
                for s in result.symptoms
            ],
            "negated": result.negated,
        }
