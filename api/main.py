"""
FastAPI entry point for AI-Analyzer.
Medical Diagnosis Pipeline by CepiaLabs.
Reference: Krish Naik - Medical Diagnosis App

Endpoints:
  POST /analyze       - Full pipeline (conversation + demographics -> risk, triage, clarification)
  POST /analyze/continue - Answer clarifications; re-run with combined context
  POST /extract       - Phase 1 only (NER extraction)
  GET  /health       - API status, LLM config
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# Fix: Uvicorn leaves custom loggers at WARNING by default, so INFO logs get filtered.
# Force root + our loggers to INFO so [NER], [FEATURES], [CLARIFICATION] etc. show
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _name in ("src.pipeline", "src.llm_reasoning", "api.main"):
    logging.getLogger(_name).setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load .env before imports (API keys)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pipeline: NER -> Ontology -> Features -> Risk Model (finetuned/SYNAPSE) -> LLM
from src.pipeline import AIAnalyzerPipeline
from src.extraction import MLNERExtractor  # ML-based (DistilBERT); use NERExtractor for rule-based
from src.llm_reasoning import LLMReasoner

app = FastAPI(title="AI-Analyzer", description="Medical Diagnosis Pipeline by CepiaLabs")
llm_reasoner = LLMReasoner()
pipeline = AIAnalyzerPipeline()
extractor = MLNERExtractor()

# Sessions: store conversation + demographics for /analyze/continue (in-memory; use Redis/DB in prod)
sessions: dict[str, dict[str, Any]] = {}
_api_logger = logging.getLogger("api.main")


@app.on_event("startup")
def startup():
    from src.llm_reasoning import LLMReasoner
    r = LLMReasoner()
    status = "configured" if r.client else "NOT configured (set OPENAI_API_KEY in .env)"
    print(f"[AI-Analyzer] LLM: {status}")


class AnalyzeRequest(BaseModel):
    conversation: str
    demographics: dict[str, Any] | None = None
    history: dict[str, Any] | None = None


class ExtractRequest(BaseModel):
    conversation: str


class ContinueRequest(BaseModel):
    session_id: str
    answers: str  # Patient/doctor answers to clarifying questions


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Run full pipeline: NER -> Ontology -> Features -> Risk -> LLM. Returns session_id for /analyze/continue."""
    _api_logger.info("[ANALYZE] conversation=%d chars, demographics=%s", len(request.conversation), request.demographics)
    result = pipeline.run(
        conversation=request.conversation,
        demographics=request.demographics,
        history=request.history,
    )
    session_id = str(uuid.uuid4())
    questions = result.get("llm_clarification", {}).get("clarifying_questions", [])
    sessions[session_id] = {
        "conversation": request.conversation,
        "demographics": request.demographics or {},
        "history": request.history or {},
        "clarifying_questions": questions,
    }
    _api_logger.info("[ANALYZE] session_id=%s, triage=%s, questions=%d", session_id,
                     result.get("triage_recommendation"), len(questions))
    result["session_id"] = session_id
    return result


@app.post("/analyze/continue")
def analyze_continue(request: ContinueRequest):
    """
    Answer clarifying questions and re-run pipeline with full context.
    Uses LLM merge: rewrites original + Q&A into a clear narrative before re-extraction.
    Fallback: simple append if LLM unavailable.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start with POST /analyze first.")
    session = sessions[request.session_id]
    questions = session.get("clarifying_questions", [])

    _api_logger.info("[CONTINUE] session=%s, answers=%d chars, stored_questions=%d",
                     request.session_id[:8], len(request.answers), len(questions))

    # LLM merge: rewrite into clear clinical narrative for better NER extraction
    if llm_reasoner.client and questions:
        _api_logger.info("[CONTINUE] Using LLM merge")
        combined = llm_reasoner.merge_clarification(
            conversation=session["conversation"],
            clarifying_questions=questions,
            patient_answers=request.answers,
        )
    else:
        _api_logger.info("[CONTINUE] Fallback: simple append (LLM=%s, questions=%d)", llm_reasoner.client, len(questions))
        combined = f"{session['conversation']}\n\nPatient clarification: {request.answers}"

    _api_logger.info("[CONTINUE] Re-running pipeline on merged text (%d chars)", len(combined))
    result = pipeline.run(
        conversation=combined,
        demographics=session.get("demographics"),
        history=session.get("history"),
    )
    _api_logger.info("[CONTINUE] Done. triage=%s, severity=%s", result.get("triage_recommendation"), result.get("severity"))
    # Update session for further follow-ups
    sessions[request.session_id]["conversation"] = combined
    sessions[request.session_id]["clarifying_questions"] = result.get("llm_clarification", {}).get("clarifying_questions", [])
    result["session_id"] = request.session_id
    return result


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear a session when done. Optional."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}


@app.post("/extract")
def extract_phase1(request: ExtractRequest):
    """Phase 1 only: NER extraction (symptoms, duration, severity, negations)."""
    result = extractor.extract(request.conversation)
    return extractor.to_dict(result)


@app.get("/health")
def health():
    """Health check: API status, LLM config, NER mode."""
    try:
        from src.llm_reasoning import LLMReasoner
        llm_ok = LLMReasoner().client is not None
    except Exception:
        llm_ok = False
    return {
        "status": "ok",
        "llm_configured": llm_ok,
        "ner_mode": "ml",
        "version": "2.1",
    }
