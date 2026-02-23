"""
Stage 5: LLM Clinical Reasoning Layer
Handles nuance, ambiguity, clarifying questions.
Guardrail: LLM cannot finalize severity - must pass through risk engine.
"""

from .reasoner import LLMReasoner

__all__ = ["LLMReasoner"]
