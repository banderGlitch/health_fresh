"""
AI-Analyzer Pipeline - End-to-End Orchestration

Flow: Conversation -> NER -> Ontology -> Features -> Risk Model -> LLM -> Final Profile
"""

from typing import Any

from .extraction import NERExtractor
from .ontology import OntologyMapper
from .features import FeatureBuilder
from .risk_model import RiskPredictor
from .llm_reasoning import LLMReasoner


class AIAnalyzerPipeline:
    """
    Orchestrates the full 7-step pipeline.
    """

    def __init__(self):
        self.extractor = NERExtractor()
        self.ontology = OntologyMapper()
        self.feature_builder = FeatureBuilder()
        self.risk_predictor = RiskPredictor()
        self.llm_reasoner = LLMReasoner()

    def run(
        self,
        conversation: str,
        demographics: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute full pipeline and return final structured symptom profile."""
        extraction_result = self.extractor.extract(conversation)
        extraction_dict = self.extractor.to_dict(extraction_result)

        mapped_symptoms = self.ontology.map_extraction_result(extraction_dict)

        features = self.feature_builder.build(
            mapped_symptoms=mapped_symptoms,
            demographics=demographics,
            history=history,
        )

        risk_output = self.risk_predictor.predict(features)

        llm_output = self.llm_reasoner.clarify(
            conversation=conversation,
            extraction_result=extraction_dict,
            risk_output=risk_output,
        )

        return {
            "symptoms": extraction_dict["symptoms"],
            "mapped_symptoms": mapped_symptoms,
            "negated": extraction_dict["negated"],
            "risk_score": risk_output["RiskScore"],
            "severity": risk_output["Severity"],
            "confidence": risk_output["Confidence"],
            "possible_conditions": risk_output["possible_conditions"],
            "llm_clarification": llm_output,
        }
