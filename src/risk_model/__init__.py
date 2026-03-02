"""
Stage 4: Risk Model

Predicts risk score, severity, possible conditions.
Uses finetuned triage (3-way) when available, SYNAPSE (2-way) as fallback,
heuristic otherwise.
"""

from .finetuned_triage_predictor import FinetunedTriagePredictor
from .predictor import RiskPredictor
from .synapse_predictor import SynapseTriagePredictor

__all__ = ["RiskPredictor", "FinetunedTriagePredictor", "SynapseTriagePredictor"]
