"""
Stage 4: Risk Model — Predict triage (OTC | Doctor | Emergency).

Tries in order:
  1. RAG triage (2-way) — if index built (semantic retrieval from SYNAPSE)
  2. Finetuned triage (3-way) — if model loaded (commented out to use RAG)
  3. SYNAPSE (2-way) — if model loaded (OTC Drug, Doctor Consultation)
  4. Heuristic — fallback using severity, red flags, syndromes
"""

import logging
import sys
from pathlib import Path
from typing import Any

from .finetuned_triage_predictor import FinetunedTriagePredictor
from .synapse_predictor import SynapseTriagePredictor

logger = logging.getLogger(__name__)


def _get_rag_predictor():
    """Lazy-load RAG triage predictor (rag_triage module). Returns None if unavailable."""
    _project_root = Path(__file__).resolve().parents[2]
    _rag_root = _project_root / "rag_triage"
    _index_dir = _rag_root / "index"
    if not _rag_root.exists():
        return None
    if not (_index_dir / "embeddings.npy").exists() or not (_index_dir / "cases.json").exists():
        logger.warning("[RISK] RAG index not found at %s — run: cd rag_triage && python build_index.py", _index_dir)
        return None
    # Add rag_triage to path so its config/imports resolve (before rag_triage.src imports)
    if str(_rag_root) not in sys.path:
        sys.path.insert(0, str(_rag_root))
    try:
        # Use rag_triage.src.triage to avoid conflict with main project's src
        from rag_triage.src.triage import RAGTriagePredictor
        from rag_triage.src.retriever import Retriever
        retriever = Retriever(index_dir=_index_dir)
        predictor = RAGTriagePredictor()
        predictor._retriever = retriever
        return predictor
    except Exception as e:
        logger.warning("[RISK] RAG load failed: %s", e)
        return None


class RiskPredictor:
    """Predict risk score and triage from features."""

    def __init__(
        self,
        use_finetuned_triage: bool = True,
        use_rag_triage: bool = True,
        use_synapse: bool = True,
    ):
        self._finetuned = FinetunedTriagePredictor() if use_finetuned_triage else None
        self._rag = None  # Lazy-loaded
        self._use_rag = use_rag_triage
        self._synapse = SynapseTriagePredictor() if use_synapse else None

    def _get_rag(self):
        if self._rag is None and self._use_rag:
            self._rag = _get_rag_predictor()
        return self._rag

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Try RAG first, then finetuned, then SYNAPSE, then heuristic."""
        # 1. RAG triage (2-way, semantic retrieval)
        rag = self._get_rag()
        if rag and rag.is_available:
            result = rag.predict(features)
            if result and result.get("triage_recommendation"):
                logger.info("[RISK] Using RAG triage")
                return result

        # 2. Finetuned triage (3-way) — commented out to use RAG
        # if self._finetuned and self._finetuned.is_available:
        #     result = self._finetuned.predict(features)
        #     if result:
        #         logger.info("[RISK] Using finetuned triage")
        #         return result

        # 3. SYNAPSE (2-way, TF-IDF)
        if self._synapse and self._synapse.is_available:
            result = self._synapse.predict(features)
            if result:
                logger.info("[RISK] Using SYNAPSE triage")
                return result

        # 4. Fallback: heuristic
        logger.info("[RISK] Using heuristic fallback")
        return self._heuristic(features)

    def _heuristic(self, features: dict[str, Any]) -> dict[str, Any]:
        """Simple rules when no model available."""
        severity = "MODERATE"
        risk = 0.5

        if features.get("has_severe") or self._worst_severity(features) == "severe":
            severity, risk = "HIGH", 0.8
        elif self._worst_severity(features) == "mild":
            severity, risk = "LOW", 0.3

        if features.get("has_red_flag"):
            risk = min(1.0, risk + 0.15)
            if features.get("red_flag_severe"):
                severity, risk = "HIGH", 0.9

        if features.get("syndrome_alarm"):
            severity, risk = "HIGH", max(risk, 0.9)

        if features.get("age_group", 0) >= 3:
            risk = min(1.0, risk + 0.05)
        if features.get("comorbidity_count", 0) >= 2:
            risk = min(1.0, risk + 0.05)

        return {
            "RiskScore": round(risk, 2),
            "Severity": severity,
            "Confidence": 0.6,
            "possible_conditions": self._conditions(features),
            "triage_recommendation": "Doctor Consultation" if risk >= 0.7 else "OTC Drug",
        }

    def _worst_severity(self, features: dict[str, Any]) -> str | None:
        sev = features.get("severities") or []
        return sev[0] if sev else None

    def _conditions(self, features: dict[str, Any]) -> list[str]:
        out = []
        if features.get("syndrome_respiratory"):
            out.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            out.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            out.append("Gastrointestinal condition")
        return out
