"""
FastAPI entry point for AI-Analyzer.
Reference: Krish Naik Medical Diagnosis App
"""

import uuid
from pathlib import Path
import sys

# Load .env before other imports (for OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from src.pipeline import AIAnalyzerPipeline
from src.extraction import NERExtractor

app = FastAPI(title="AI-Analyzer", description="Medical Diagnosis Pipeline by CepiaLabs")
pipeline = AIAnalyzerPipeline()
extractor = NERExtractor()

# In-memory session store: session_id -> { conversation, demographics, history }
# For production, use Redis or a database
sessions: dict[str, dict[str, Any]] = {}


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
    """Run full pipeline. Returns session_id for follow-up answers."""
    result = pipeline.run(
        conversation=request.conversation,
        demographics=request.demographics,
        history=request.history,
    )
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "conversation": request.conversation,
        "demographics": request.demographics or {},
        "history": request.history or {},
    }
    result["session_id"] = session_id
    return result


@app.post("/analyze/continue")
def analyze_continue(request: ContinueRequest):
    """
    Answer clarifying questions and re-run pipeline with full context.
    Combines original conversation + your answers, then re-extracts and re-analyzes.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start with POST /analyze first.")
    session = sessions[request.session_id]
    # Append answers to conversation (maintains full context)
    combined = f"{session['conversation']}\n\nPatient clarification: {request.answers}"
    result = pipeline.run(
        conversation=combined,
        demographics=session.get("demographics"),
        history=session.get("history"),
    )
    # Update stored conversation for further follow-ups
    sessions[request.session_id]["conversation"] = combined
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
    """Health check + LLM status."""
    try:
        from src.llm_reasoning import LLMReasoner
        r = LLMReasoner()
        llm_ok = r.client is not None
    except Exception:
        llm_ok = False
    return {
        "status": "ok",
        "llm_configured": llm_ok,
        "version": "2.0",
    }
