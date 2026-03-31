"""
Stage 1: Extract symptoms from conversation (rule-based).
Symptoms from lexicon; duration, severity, negation from regex.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .symptom_lexicon import SYMPTOM_LEXICON, get_canonical


@dataclass
class ExtractedSymptom:
    name: str
    duration: str | None = None
    severity: str | None = None
    associated_factors: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    symptoms: list[ExtractedSymptom]
    negated: list[str]
    help_message: str | None = None


_SEV = [
    ("severe", re.compile(r"\b(severe|intense|extreme|bad|terrible|sharp|acute)\b", re.I)),
    ("moderate", re.compile(r"\b(moderate|moderately|medium)\b", re.I)),
    ("mild", re.compile(r"\b(mild|slight|minor|a bit|little)\b", re.I)),
]

_DUR = re.compile(
    r"(?:for|since|lasting|about|over)\s+(\d+\s*(?:days?|weeks?|months?|hours?)|a\s+(?:day|week|month|hour|few\s+days)|(?:yesterday|today|last\s+night|this\s+morning))|\b(\d+\s*(?:days?|weeks?|months?|hours?)|(?:a\s+)?few\s+days)\b",
    re.I,
)

_NEG = re.compile(r"\b(no|not|without|denies?)\s+", re.I)

_NEG_PHRASE = [
    re.compile(r"\bno\s+([a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    re.compile(r"\b(?:not\s+(?:having|experiencing|had|have)|without|denies?)\s+([a-zA-Z\s]+?)(?=\.|,|$)", re.I),
]


def _overlaps(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 < e2 and s2 < e1


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def to_result_dict(result: ExtractionResult) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    if result.help_message:
        out["help_message"] = result.help_message
    return out


class NERExtractor:
    """Extract symptoms via lexicon + regex. Used by pipeline and MLNERExtractor."""

    def __init__(self):
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        for name, variations in SYMPTOM_LEXICON.items():
            for variation in variations:
                patterns.append((name, re.compile(rf"\b{re.escape(variation)}\b", re.I)))
        patterns.sort(key=lambda x: -len(x[1].pattern))
        return patterns

    def _extract_negations(self, text: str) -> list[str]:
        negated: list[str] = []
        for pat in _NEG_PHRASE:
            for match in pat.finditer(text):
                phrase = match.group(1).strip()
                for part in re.split(r"\s+or\s+|\s+and\s+", phrase, flags=re.I):
                    cleaned = part.strip()
                    if len(cleaned) <= 2:
                        continue
                    canonical = get_canonical(cleaned) or cleaned
                    if canonical not in negated:
                        negated.append(canonical)
        return negated

    def _extract_duration_near(self, text: str, start: int, end: int, window: int = 80) -> str | None:
        ctx = text[max(0, start - window) : min(len(text), end + window)]
        match = _DUR.search(ctx)
        return (match.group(1) or match.group(2) or "").strip() if match else None

    def _extract_severity_near(self, text: str, start: int, _end: int, window: int = 50) -> str | None:
        ctx = text[max(0, start - window) : start]
        best, best_pos = None, -1
        for name, pat in _SEV:
            for match in pat.finditer(ctx):
                if match.end() > best_pos:
                    best_pos = match.end()
                    best = name
        return best

    def _collect_symptom_spans(self, text: str) -> list[tuple[str, int, int]]:
        found: list[tuple[str, int, int]] = []
        for name, pat in self._patterns:
            for match in pat.finditer(text):
                start, end = match.start(), match.end()
                if not any(_overlaps(start, end, s2, e2) for _, s2, e2 in found):
                    found.append((name, start, end))
        found.sort(key=lambda x: x[1])
        return found

    def extract_negations(self, text: str) -> list[str]:
        return self._extract_negations(text)

    def _is_negated_match(self, text: str, start: int, name: str, negated_lower: set[str]) -> bool:
        if name.lower() in negated_lower:
            return True
        return bool(_NEG.search(text[max(0, start - 50) : start]))

    def _associated_symptoms(self, name: str, found: list[tuple[str, int, int]], negated_lower: set[str]) -> list[str]:
        return [other for other, _, _ in found if other != name and other.lower() not in negated_lower]

    def build_symptoms_from_spans(
        self,
        text: str,
        found: list[tuple[str, int, int]],
        negated: list[str],
    ) -> list[ExtractedSymptom]:
        negated_lower = {n.lower() for n in negated}
        symptoms: list[ExtractedSymptom] = []

        for name, start, end in found:
            if self._is_negated_match(text, start, name, negated_lower):
                continue
            symptoms.append(
                ExtractedSymptom(
                    name=name,
                    duration=self._extract_duration_near(text, start, end),
                    severity=self._extract_severity_near(text, start, end),
                    associated_factors=self._associated_symptoms(name, found, negated_lower),
                )
            )
        return symptoms

    def extract(self, text: str) -> ExtractionResult:
        normalized_text = normalize_text(text)
        negated = self._extract_negations(normalized_text)
        found = self._collect_symptom_spans(normalized_text)
        symptoms = self.build_symptoms_from_spans(normalized_text, found, negated)
        return ExtractionResult(symptoms=symptoms, negated=negated)

    def to_dict(self, result: ExtractionResult) -> dict[str, Any]:
        return to_result_dict(result)
