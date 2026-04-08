"""
Lightweight "RAG" for symptoms: load phrases from the SYNAPSE CSV, score by word overlap
with the patient text, return substring spans. No ML deps — simple and readable.

Phrases are loaded once and cached. Edit CSV path only if your file moves.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Same default as scripts/sync_synapse_lexicon.py
_DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv"
)

_SKIP = frozenset(
    {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "red", "white"}
)
_BAD = frozenset({"a white"})

_phrases_cache: list[str] | None = None


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def load_phrases(csv_path: Path | None = None) -> list[str]:
    """Unique normalized symptom phrases from the Symptoms column."""
    global _phrases_cache
    if _phrases_cache is not None:
        return _phrases_cache

    path = csv_path or _DEFAULT_CSV
    if not path.is_file():
        _phrases_cache = []
        return _phrases_cache

    seen: set[str] = set()
    out: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Symptoms") or ""
            for part in raw.split(","):
                key = " ".join(part.strip().lower().split())
                if len(key) < 3 or key in _SKIP or key in _BAD:
                    continue
                if len(key) <= 6 and key.startswith("a "):
                    continue
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    _phrases_cache = out
    return _phrases_cache


def rag_spans(text: str, top_k: int = 15, min_overlap: float = 0.34) -> list[tuple[str, int, int]]:
    """
    Return (phrase, start, end) for phrases that appear in text and score well by overlap.

    Overlap = |phrase_words ∩ query_words| / |phrase_words|  (phrase recall in query).
    """
    if not text or not text.strip():
        return []

    phrases = load_phrases()
    if not phrases:
        return []

    tl = text.lower()
    q_tokens = _tokenize(tl)
    scored: list[tuple[float, int, str]] = []

    for p in phrases:
        if p not in tl:
            continue
        pt = _tokenize(p)
        if not pt:
            continue
        overlap = len(pt & q_tokens) / len(pt)
        if overlap < min_overlap:
            continue
        scored.append((overlap, len(p), p))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

    seen_starts: set[int] = set()
    out: list[tuple[str, int, int]] = []
    for _, _, p in scored:
        if len(out) >= top_k:
            break
        i = tl.find(p)
        if i < 0:
            continue
        if i in seen_starts:
            continue
        seen_starts.add(i)
        out.append((p, i, i + len(p)))

    return out


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "I have fever and headache"
    print("Phrases:", len(load_phrases()))
    print("Query:", q)
    print("Spans:", rag_spans(q))
