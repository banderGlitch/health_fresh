"""
Stage 4: Probabilistic Inference Model
Predicts: likely conditions, severity risk, confidence.
Model types: Gradient boosting, Bayesian network.
"""

from .predictor import RiskPredictor

__all__ = ["RiskPredictor"]
