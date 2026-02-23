"""
Stage 1: Structured NLP Extraction
- Medical NER for symptoms, duration, severity
- Pattern rules for units & time
- Negation detection
"""

from .ner_extractor import NERExtractor

__all__ = ["NERExtractor"]
