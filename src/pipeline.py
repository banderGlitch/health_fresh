"""
AI-Analyzer Pipeline - End-to-End Orchestration

Flow: Conversation -> NER -> Ontology -> Features -> Risk Model -> LLM -> Final Profile

Each request passes through 5 stages:
  1. NER: Extract symptoms, duration, severity, negations from free text
  2. Ontology: Map symptoms to SNOMED codes for standardisation
  3. Features: Build numeric/categorical features for the risk model
  4. Risk Model: Finetuned triage (OTC|Doctor|Emergency) or SYNAPSE fallback
  5. LLM: Generate clarification questions for the patient
"""

import json
import logging
from typing import Any

from .extraction import MLNERExtractor
from .ontology import OntologyMapper    # Phase 2: SNOMED mapping
from .features import FeatureBuilder    # Phase 3: feature vectors
from .risk_model import RiskPredictor   # Phase 4: risk scoring
from .llm_reasoning import LLMReasoner # Phase 5: clarification

logger = logging.getLogger(__name__)


def _log_features(features: dict[str, Any]) -> None:
    """Log key features for real-time debugging. Skips large/verbose fields."""
    key_fields = {
        "symptom_text",
        "symptom_count",
        "max_severity",
        "has_severe",
        "has_red_flag",
        "red_flag_count",
        "max_duration_days",
        "acute_flag",
        "age",
        "age_group",
        "gender",
        "syndrome_cardiac_like",
        "syndrome_respiratory",
        "syndrome_gi",
        "syndrome_alarm",
    }
    subset = {k: v for k, v in features.items() if k in key_fields}
    logger.info("[FEATURES] %s", json.dumps(subset, default=str))


class AIAnalyzerPipeline:
    """
    Orchestrates the full pipeline. Uses MLNERExtractor by default.
    """

    def __init__(self):
        # Phase 1: fixed ML-based NER extractor
        self.ner_mode = "ml"
        self.extractor = MLNERExtractor()
        # Phase 2: Map symptom names to SNOMED CT codes
        self.ontology = OntologyMapper()
        # Phase 3: Build feature dict (symptom counts, severity, red flags, demographics)
        self.feature_builder = FeatureBuilder()
        # Phase 4: Finetuned triage (3-way) or SYNAPSE (2-way); heuristic fallback
        self.risk_predictor = RiskPredictor()
        # Phase 5: LLM (OpenAI/Groq/Gemini) for clarification questions
        self.llm_reasoner = LLMReasoner()

    def run(
        self,
        conversation: str,
        demographics: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
        patient_grounding_text: str | None = None,
    ) -> dict[str, Any]:
        """Execute full pipeline and return final structured symptom profile.

        patient_grounding_text: if set, symptoms must appear in this text (e.g. original message + answers).
        Use when ``conversation`` is an LLM-merged narrative so invented symptoms are dropped.
        """
        # --- Phase 1: NER extraction ---
        # Parse conversation; extract symptoms with name, duration, severity; detect negations (e.g. "no fever")
        extraction_result = self.extractor.extract(conversation)
        extraction_dict = self.extractor.to_dict(extraction_result)
        if patient_grounding_text:
            from .extraction.symptom_lexicon import ground_extraction_dict

            extraction_dict = ground_extraction_dict(extraction_dict, patient_grounding_text)

        # Real-time log: NER extraction
        symptoms_list = extraction_dict.get("symptoms") or []
        negated_list = extraction_dict.get("negated") or []
        
        logger.info(
            "[NER] Extracted %d symptoms: %s",
            len(symptoms_list),
            [(s.get("name"), s.get("duration"), s.get("severity")) for s in symptoms_list],
        )
        logger.info("[NER] Negated: %s", negated_list)

        # --- Phase 2: Ontology mapping ---
        # Assign SNOMED codes to each symptom; unmapped symptoms kept with name only
        mapped_symptoms = self.ontology.map_extraction_result(extraction_dict)

        # Real-time log: ontology mapping
        logger.info(
            "[ONTOLOGY] Mapped %d symptoms: %s",
            len(mapped_symptoms),
            [(m.get("name"), m.get("snomed_code")) for m in mapped_symptoms],
        )

        # --- Phase 3: Feature building ---
        # Compute symptom_text, severity flags, red flags, syndromes, age/gender, comorbidity count
        features = self.feature_builder.build(
            mapped_symptoms=mapped_symptoms,
            extraction_dict=extraction_dict,  # Full symptoms (incl. unmapped)
            demographics=demographics,
            history=history,
            conversation=conversation,
        )

        # Real-time log: features (key fields for debugging)
        _log_features(features)

        # --- Phase 4: Risk prediction ---
        # Finetuned triage (OTC|Doctor|Emergency) or SYNAPSE (OTC|Doctor); heuristic fallback
        risk_output = self.risk_predictor.predict(features)

        # Real-time log: prediction
        logger.info(
            "[PREDICTION] RiskScore=%.2f | Severity=%s | Triage=%s | Confidence=%.2f",
            risk_output.get("RiskScore", 0),
            risk_output.get("Severity", ""),
            risk_output.get("triage_recommendation", "N/A"),
            risk_output.get("Confidence", 0),
        )
        if risk_output.get("possible_conditions"):
            logger.info("[PREDICTION] Possible conditions: %s", risk_output["possible_conditions"])

        # --- Phase 5: LLM clarification ---
        # Generate follow-up questions based on symptoms and risk
        llm_output = self.llm_reasoner.clarify(
            conversation=conversation,
            extraction_result=extraction_dict,
            risk_output=risk_output,
        )

        # Assemble final response for API
        result: dict[str, Any] = {
            "symptoms": extraction_dict["symptoms"],
            "mapped_symptoms": mapped_symptoms,
            "negated": extraction_dict["negated"],
            "risk_score": risk_output["RiskScore"],
            "severity": risk_output["Severity"],
            "confidence": risk_output["Confidence"],
            "possible_conditions": risk_output["possible_conditions"],
            "llm_clarification": llm_output,
        }
        if "triage_recommendation" in risk_output and risk_output["triage_recommendation"]:
            result["triage_recommendation"] = risk_output["triage_recommendation"]
        if extraction_dict.get("help_message"):
            result["help_message"] = extraction_dict["help_message"]
        return result
