"""
Stage 2: Ontology Mapping Layer
- Symptom normalization
- Synonym expansion
- SNOMED CT / medical vocabulary mapping
"""

from .mapper import OntologyMapper

__all__ = ["OntologyMapper"]
