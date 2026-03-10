"""
BACKUP - Original ner_extractor.py (before simplification)
Restore by copying this file back to ner_extractor.py
"""

import re
from typing import Any
from dataclasses import dataclass, field

from .symptom_lexicon import SYMPTOM_LEXICON, VARIATION_TO_CANONICAL, get_canonical


@dataclass
class ExtractedSymptom:
    """Single extracted symptom with attributes."""
    name: str
    duration: str | None = None
    severity: str | None = None  # mild | moderate | severe
    associated_factors: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Output of Stage 1 - matches pipeline spec."""
    symptoms: list[ExtractedSymptom]
    negated: list[str]


class NERExtractor:
    """
    Phase 1: Extracts structured medical facts from conversation.
    - Symptoms (lexicon-based)
    - Duration (pattern rules)
    - Severity (mild/moderate/severe)
    - Associated factors
    - Negations
    """

    SEVERITY_PATTERNS = {
        "mild": re.compile(r"\b(mild|slight|minor|a bit|little)\b", re.IGNORECASE),
        "moderate": re.compile(r"\b(moderate|moderately|medium|moderately)\b", re.IGNORECASE),
        "severe": re.compile(r"\b(severe|intense|extreme|bad|terrible|sharp|acute)\b", re.IGNORECASE),
    }

    DURATION_PATTERN = re.compile(
        r"(?:for|since|lasting|about|over)\s+"
        r"(\d+\s*(?:days?|weeks?|months?|hours?|minutes?)|"
        r"a\s+(?:day|week|month|hour|few\s+days)|"
        r"(?:yesterday|today|last\s+night|this\s+morning))",
        re.IGNORECASE
    )

    DURATION_STANDALONE = re.compile(
        r"\b(\d+\s*(?:days?|weeks?|months?|hours?|minutes?)|"
        r"(?:a\s+)?few\s+days)\b",
        re.IGNORECASE
    )

    NEGATION_PHRASES = [
        r"\bno\s+([a-zA-Z\s]+?)(?=\.|,|$)",
        r"\bnot\s+(?:having|experiencing|had|have)\s+([a-zA-Z\s]+?)(?=\.|,|$)",
        r"\bwithout\s+([a-zA-Z\s]+?)(?=\.|,|$)",
        r"\bdenies?\s+([a-zA-Z\s]+?)(?=\.|,|$)",
        r"\bnever\s+(?:had|experienced)\s+([a-zA-Z\s]+?)(?=\.|,|$)",
        r"\b(?:doesn't|don't|haven't|hasn't)\s+(?:have|had)\s+([a-zA-Z\s]+?)(?=\.|,|$)",
    ]

    def __init__(self):
        self._symptom_patterns = self._build_symptom_patterns()

    def _build_symptom_patterns(self) -> list[tuple[str, re.Pattern]]:
        patterns = []
        for canonical, variations in SYMPTOM_LEXICON.items():
            for v in variations:
                escaped = re.escape(v)
                patterns.append((canonical, re.compile(rf"\b{escaped}\b", re.IGNORECASE)))
        patterns.sort(key=lambda x: -len(x[1].pattern))
        return patterns

    def _extract_negations(self, text: str) -> list[str]:
        negated = []
        for pattern_str in self.NEGATION_PHRASES:
            for m in re.finditer(pattern_str, text, re.IGNORECASE):
                phrase = m.group(1).strip()
                if len(phrase) > 2:
                    for part in re.split(r"\s+or\s+|\s+and\s+", phrase, flags=re.IGNORECASE):
                        part = part.strip()
                        if len(part) > 2:
                            canonical = get_canonical(part) or part
                            if canonical not in negated:
                                negated.append(canonical)
        return negated

    def _extract_duration_near(self, text: str, start: int, end: int, window: int = 80) -> str | None:
        left = max(0, start - window)
        right = min(len(text), end + window)
        context = text[left:right]
        m = self.DURATION_PATTERN.search(context)
        if m:
            return m.group(1).strip()
        m = self.DURATION_STANDALONE.search(context)
        if m:
            return m.group(1).strip()
        return None

    def _extract_severity_near(self, text: str, start: int, end: int, window: int = 50) -> str | None:
        pre_start = max(0, start - window)
        pre_context = text[pre_start:start]
        best_sev, best_pos = None, -1
        for sev, pat in self.SEVERITY_PATTERNS.items():
            for m in pat.finditer(pre_context):
                pos = pre_start + m.end()
                if pos > best_pos:
                    best_pos, best_sev = pos, sev
        return best_sev

    def _get_associated_symptoms(self, symptom_name: str, all_found: list[tuple[str, int, int]]) -> list[str]:
        associated = []
        for name, _, _ in all_found:
            if name != symptom_name and name not in associated:
                associated.append(name)
        return associated

    def extract(self, text: str) -> ExtractionResult:
        text_clean = " ".join(text.split())
        negated = self._extract_negations(text_clean)

        found: list[tuple[str, int, int]] = []
        for canonical, pattern in self._symptom_patterns:
            for m in pattern.finditer(text_clean):
                span_start, span_end = m.span()
                overlap = any(s <= span_start < e or s < span_end <= e for _, s, e in found)
                if not overlap:
                    found.append((canonical, span_start, span_end))

        found.sort(key=lambda x: x[1])
        merged: list[tuple[str, int, int]] = []
        for name, s, e in found:
            if not any(ms <= s < me or ms < e <= me for _, ms, me in merged):
                merged.append((name, s, e))

        negated_set = {n.lower() for n in negated}
        symptoms: list[ExtractedSymptom] = []

        for name, start, end in merged:
            if name.lower() in negated_set:
                continue
            left_context = text_clean[max(0, start - 50):start].lower()
            if re.search(r"\b(no|not|without|denies?)\s+", left_context):
                continue

            duration = self._extract_duration_near(text_clean, start, end)
            severity = self._extract_severity_near(text_clean, start, end)
            associated = [
                a for a in self._get_associated_symptoms(name, merged)
                if a.lower() not in negated_set
            ]

            symptoms.append(ExtractedSymptom(
                name=name,
                duration=duration,
                severity=severity,
                associated_factors=associated,
            ))

        return ExtractionResult(symptoms=symptoms, negated=negated)

    def to_dict(self, result: ExtractionResult) -> dict[str, Any]:
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
