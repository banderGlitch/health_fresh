"""
Stage 1: Structured NLP Extraction
- ML: in-house trained DistilBERT (symptom spans)
- Rule-based: duration, severity, negation (post-processing)
- Fallback: NERExtractor (rule-based, lexicon+regex)
"""

from .ml_ner_extractor import MLNERExtractor
from .ner_extractor import NERExtractor  # Rule-based; comment in pipeline/api to use

__all__ = ["MLNERExtractor", "NERExtractor"]
