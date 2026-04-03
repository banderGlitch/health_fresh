"""
FastAPI: AI-Analyzer medical triage pipeline.
POST /analyze, POST /analyze/continue, POST /extract, GET /health
"""

import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# Logging: flush immediately (helps Windows)
class _FlushHandler(logging.StreamHandler):
    def emit(self, r):
        super().emit(r)
        self.flush()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_FlushHandler(sys.stdout)], force=True)
for n in ("src.pipeline", "src.risk_model", "src.llm_reasoning", "api.main", "uvicorn", "uvicorn.access"):
    logging.getLogger(n).setLevel(logging.INFO)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# This allows Python to find your src folder.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import AIAnalyzerPipeline
from src.extraction import MLNERExtractor
from src.llm_reasoning import LLMReasoner
try:
    from src.storage import MongoSessionStore
except Exception:
    MongoSessionStore = None  # type: ignore[assignment]

app = FastAPI(title="AI-Analyzer", description="Medical triage pipeline")
log = logging.getLogger("api.main")
pipeline = AIAnalyzerPipeline()
llm = LLMReasoner()
extractor = MLNERExtractor()
# Local in-memory fallback (kept for reference, intentionally disabled):
# sessions: dict[str, dict[str, Any]] = {}
session_store = None
if MongoSessionStore:
    try:
        session_store = MongoSessionStore()
    except Exception as e:
        log.warning("Mongo session store unavailable (in-memory fallback disabled): %s", e)


def _session_backend_name() -> str:
    return "mongodb" if session_store else "mongodb-unavailable"

MAX_FOLLOWUP_QUESTIONS = 2  # Cap at 2; LLM instructed to ask 1-2 only
MAX_ROUNDS = 4
_DEMOGRAPHIC_QUESTIONS = {
    "age": "What is your age?",
    "gender": "What is your gender (male/female)?",
}
_DEMOGRAPHIC_REASK = {
    "age": "I still need your age to continue triage. Please share it in years.",
    "gender": "I still need your gender (male/female) to continue triage.",
}
_EMERGENCY_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "breathing difficulty",
    "coughing blood",
    "blood in stool",
}


def _create_session(session_id: str, payload: dict[str, Any]) -> None:
    if session_store:
        session_store.create_session(session_id, payload)
        return
    # Local fallback disabled on purpose:
    # sessions[session_id] = payload
    raise RuntimeError("MongoDB session store is not available")


def _get_session(session_id: str) -> dict[str, Any] | None:
    if session_store:
        return session_store.get_session(session_id)
    # Local fallback disabled on purpose:
    # return sessions.get(session_id)
    raise RuntimeError("MongoDB session store is not available")


def _update_session(session_id: str, payload: dict[str, Any]) -> None:
    if session_store:
        session_store.update_session(session_id, payload)
        return
    # Local fallback disabled on purpose:
    # sessions[session_id] = payload
    raise RuntimeError("MongoDB session store is not available")


def _delete_session(session_id: str) -> bool:
    if session_store:
        return session_store.delete_session(session_id)
    # Local fallback disabled on purpose:
    # if session_id in sessions:
    #     del sessions[session_id]
    #     return True
    return False


def _extract_demographics_from_text(text: str, existing: dict[str, Any]) -> dict[str, Any]:
    """Very small parser for age/gender from free-text answers."""
    out = dict(existing or {})
    t = (text or "").lower()

    age_match = re.search(r"\b(?:i am|i'm|age is|age)\s*(\d{1,3})\b", t) or re.search(r"\b(\d{1,3})\s*(?:years?|yrs?)\b", t)
    if age_match:
        try:
            age = int(age_match.group(1))
            if 0 < age <= 120:
                out["age"] = age
        except (TypeError, ValueError):
            pass

    if re.search(r"\b(male|man|boy|m)\b", t):
        out["gender"] = "male"
    elif re.search(r"\b(female|woman|girl|f)\b", t):
        out["gender"] = "female"

    return out


def _missing_demographic_questions(demographics: dict[str, Any], asked_questions: list[str]) -> list[str]:
    _ = asked_questions
    out: list[str] = []
    for key, q in _DEMOGRAPHIC_QUESTIONS.items():
        if demographics.get(key):
            continue
        out.append(q)
    return out


def _missing_required_fields(demographics: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not demographics.get("age"):
        missing.append("age")
    if not demographics.get("gender"):
        missing.append("gender")
    return missing


def _next_questions(llm_questions: list[str], demographics: dict[str, Any], asked_questions: list[str]) -> list[str]:
    """Build next turn questions: demographics first, then LLM; no repeats; max N."""
    result: list[str] = []
    seen = {q.lower().strip() for q in asked_questions}

    for q in _missing_demographic_questions(demographics, asked_questions):
        norm = q.lower().strip()
        if norm in seen:
            continue
        result.append(q)
        seen.add(norm)
        if len(result) >= MAX_FOLLOWUP_QUESTIONS:
            return result

    for q in (llm_questions or []):
        if not q or not isinstance(q, str):
            continue
        norm = q.lower().strip()
        if not norm or norm in seen:
            continue
        result.append(q.strip())
        seen.add(norm)
        if len(result) >= MAX_FOLLOWUP_QUESTIONS:
            break
    return result


def _is_emergency_outcome(result: dict[str, Any]) -> bool:
    triage = str(result.get("triage_recommendation", "")).lower()
    if triage == "emergency":
        return True
    severity = str(result.get("severity", "")).upper()
    risk = float(result.get("risk_score", 0) or 0)
    return severity == "HIGH" and risk >= 0.9


def _has_emergency_signal(extraction_dict: dict[str, Any]) -> bool:
    symptoms = extraction_dict.get("symptoms") or []
    for s in symptoms:
        name = str(s.get("name", "")).lower().strip()
        severity = str(s.get("severity", "")).lower().strip()
        if name in _EMERGENCY_SYMPTOMS:
            return True
        if severity == "severe" and ("chest" in name or "breath" in name):
            return True
    return False


def _ready_for_full_pipeline(extraction_dict: dict[str, Any], demographics: dict[str, Any]) -> bool:
    """Simple gate: enough symptom detail. Demographics improve quality but are optional."""
    _ = demographics
    symptoms = extraction_dict.get("symptoms") or []
    if not symptoms:
        return False
    has_attr = any((s.get("duration") or s.get("severity")) for s in symptoms)
    return bool(has_attr or len(symptoms) >= 2)


def _collection_llm_questions(conversation: str, extraction_dict: dict[str, Any]) -> list[str]:
    """Use LLM for follow-up questions before full pipeline is run."""
    if not llm.client:
        return []
    provisional_risk = {
        "RiskScore": 0.5,
        "Severity": "MODERATE",
        "Confidence": 0.5,
        "possible_conditions": [],
        "triage_recommendation": "",
    }
    out = llm.clarify(
        conversation=conversation,
        extraction_result=extraction_dict,
        risk_output=provisional_risk,
    )
    return out.get("clarifying_questions", []) or []


def _classify_and_get_collection_response(
    conversation: str,
    extraction_dict: dict[str, Any],
    demographics: dict[str, Any],
    asked_questions: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """
    Use LLM to classify extraction gap (vague vs unrecognized vs partial).
    Returns (response_dict, next_questions).
    - vague: clarifying_questions only, no help_message
    - unrecognized/partial: help_message only, no clarifying_questions
    - normal: standard clarifying_questions
    """
    gap = llm.classify_extraction_gap(conversation, extraction_dict)
    scenario = gap.get("scenario", "normal")

    if scenario in ("unrecognized", "partial"):
        out = _build_collection_response(
            extraction_dict=extraction_dict,
            next_qs=[],
            reason_summary="Collecting more details before full risk scoring.",
        )
        out["help_message"] = gap.get("help_message") or (
            "We couldn't recognize some terms. Could you describe your symptoms again using simpler words (e.g. sweating, fever, headache)?"
        )
        return out, []

    if scenario == "vague":
        out = _build_collection_response(
            extraction_dict=extraction_dict,
            next_qs=[],
            reason_summary="Collecting more details before full risk scoring.",
        )
        if "help_message" in out:
            del out["help_message"]
        llm_qs = gap.get("clarifying_questions") or ["Can you be more specific? What symptoms are you experiencing?"]
        next_qs = _next_questions(llm_qs, demographics, asked_questions)
        return out, next_qs

    llm_qs = _collection_llm_questions(conversation, extraction_dict)
    out = _build_collection_response(
        extraction_dict=extraction_dict,
        next_qs=[],
        reason_summary="Collecting more details before full risk scoring.",
    )
    next_qs = _next_questions(llm_qs, demographics, asked_questions)
    return out, next_qs


def _build_collection_response(
    extraction_dict: dict[str, Any],
    next_qs: list[str],
    reason_summary: str = "",
) -> dict[str, Any]:
    """Response shape used while collecting info (before full risk run)."""
    out: dict[str, Any] = {
        "symptoms": extraction_dict.get("symptoms", []),
        "mapped_symptoms": [],
        "negated": extraction_dict.get("negated", []),
        "risk_score": None,
        "severity": None,
        "confidence": None,
        "possible_conditions": [],
        "llm_clarification": {
            "clarifying_questions": next_qs,
            "reasoning_summary": reason_summary,
            "risk_output": {},
        },
        "collection_only": True,
    }
    if extraction_dict.get("help_message"):
        out["help_message"] = extraction_dict["help_message"]
    return out


def _build_round_result(round_no: int, out: dict[str, Any]) -> dict[str, Any]:
    """Small round snapshot for persistence/audit."""
    return {
        "round": round_no,
        "collection_only": bool(out.get("collection_only", False)),
        "risk_score": out.get("risk_score"),
        "severity": out.get("severity"),
        "confidence": out.get("confidence"),
        "triage_recommendation": out.get("triage_recommendation"),
        "possible_conditions": out.get("possible_conditions", []),
        "conversation_status": out.get("conversation_status"),
        "conversation_status_reason": out.get("conversation_status_reason"),
    }


def _conversation_status(round_no: int, next_qs: list[str], result: dict[str, Any], required_missing: bool) -> tuple[str, str]:
    """Return (status, reason) for frontend-friendly flow control."""
    if _is_emergency_outcome(result):
        return "completed", "emergency_detected"
    if required_missing and round_no < MAX_ROUNDS and next_qs:
        return "collecting", "required_info_missing"
    if round_no >= MAX_ROUNDS:
        return "completed", "max_rounds_reached"
    if not next_qs:
        return "completed", "no_more_questions"
    return "collecting", "followup_needed"


@app.on_event("startup")
def startup():
    print(f"[AI-Analyzer] LLM: {'configured' if llm.client else 'NOT configured (set API key in .env)'}")
    backend = _session_backend_name()
    print(f"[AI-Analyzer] Session store: {backend}")
    log.info("[SESSION_STORE] Active backend: %s", backend)


class AnalyzeRequest(BaseModel):
    conversation: str
    demographics: dict[str, Any] | None = None
    history: dict[str, Any] | None = None


class ExtractRequest(BaseModel):
    conversation: str


class ContinueRequest(BaseModel):
    session_id: str
    answers: str


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Full pipeline: NER -> Ontology -> Features -> Risk -> LLM."""
    log.info("[ANALYZE] %d chars", len(req.conversation))
    log.info("[SESSION_STORE] /analyze using backend=%s", _session_backend_name())
    initial_demographics = req.demographics or {}

    # Step 1: collection pass (extract symptoms first, then decide whether full pipeline is needed).
    extraction_result = extractor.extract(req.conversation)
    extraction_dict = extractor.to_dict(extraction_result)
    emergency = _has_emergency_signal(extraction_dict)
    ready = _ready_for_full_pipeline(extraction_dict, initial_demographics)

    force_run = False
    if emergency or ready or force_run:
        out = pipeline.run(
            conversation=req.conversation,
            demographics=initial_demographics,
            history=req.history,
        )
        llm_qs = out.get("llm_clarification", {}).get("clarifying_questions", [])
        next_qs = _next_questions(llm_qs, initial_demographics, asked_questions=[])
    else:
        out, next_qs = _classify_and_get_collection_response(
            req.conversation, extraction_dict, initial_demographics, asked_questions=[],
        )

    sid = str(uuid.uuid4())
    missing_required = _missing_required_fields(initial_demographics)
    session_payload = {
        "conversation": req.conversation,
        "demographics": initial_demographics,
        "history": req.history or {},
        "clarifying_questions": next_qs,
        "asked_questions": list(next_qs),
        "round": 1,
        "round_results": [],
    }
    status, reason = _conversation_status(1, next_qs, out, required_missing=bool(missing_required))
    session_payload["status"] = status
    session_payload["status_reason"] = reason
    _create_session(sid, session_payload)
    if "llm_clarification" in out and isinstance(out["llm_clarification"], dict):
        out["llm_clarification"]["clarifying_questions"] = next_qs
    out["session_id"] = sid
    out["conversation_status"] = status
    out["conversation_status_reason"] = reason
    out["conversation_round"] = 1
    out["max_rounds"] = MAX_ROUNDS
    out["ready_for_full_pipeline"] = bool(ready or emergency or force_run)
    out["emergency_signal_detected"] = bool(emergency)
    session_payload["round_results"].append(_build_round_result(1, out))
    session_payload["latest_risk_output"] = {
        "risk_score": out.get("risk_score"),
        "severity": out.get("severity"),
        "confidence": out.get("confidence"),
        "triage_recommendation": out.get("triage_recommendation"),
        "possible_conditions": out.get("possible_conditions", []),
    }
    _update_session(sid, session_payload)
    log.info("[ANALYZE] session=%s triage=%s", sid[:8], out.get("triage_recommendation"))
    out["patient_conversation"] = req.conversation
    return out

# clarifying_questions = active queue
# asked_questions = full asked log / memory

@app.post("/analyze/continue")
def analyze_continue(req: ContinueRequest):
    """Merge Q&A with original, re-run pipeline."""
    log.info("[CONTINUE] session=%s answers=%d chars", req.session_id[:8] if req.session_id else "?", len(req.answers))
    log.info("[SESSION_STORE] /analyze/continue using backend=%s", _session_backend_name())
    s = _get_session(req.session_id)
    if not s:
        raise HTTPException(404, "Session not found. Start with POST /analyze first.")
    s["round"] = int(s.get("round", 1)) + 1
    qs = s.get("clarifying_questions", [])
    
    if llm.client and qs:
        combined = llm.merge_clarification(s["conversation"], qs, req.answers)
    else:
        combined = f"{s['conversation']}\n\nPatient clarification: {req.answers}"

    # Update demographics from latest patient text when possible.
    updated_demographics = _extract_demographics_from_text(req.answers, s.get("demographics") or {})
    s["demographics"] = updated_demographics

    # Collection pass on merged conversation.
    extraction_result = extractor.extract(combined)
    extraction_dict = extractor.to_dict(extraction_result)
    emergency = _has_emergency_signal(extraction_dict)
    ready = _ready_for_full_pipeline(extraction_dict, updated_demographics)
    force_run = s["round"] >= MAX_ROUNDS

    asked_questions = s.get("asked_questions", [])

    if emergency or ready or force_run:
        out = pipeline.run(
            conversation=combined,
            demographics=updated_demographics,
            history=s.get("history"),
        )
        llm_qs = out.get("llm_clarification", {}).get("clarifying_questions", [])
        next_qs = [] if s["round"] >= MAX_ROUNDS else _next_questions(llm_qs, updated_demographics, asked_questions)
    else:
        out, next_qs = _classify_and_get_collection_response(
            combined, extraction_dict, updated_demographics, asked_questions,
        )
        if s["round"] >= MAX_ROUNDS:
            next_qs = []

    s["conversation"] = combined

    # If required demographics are still missing and no new question is available,
    # re-ask missing demographic prompts (unless we're showing help_message only).
    missing_required = _missing_required_fields(updated_demographics)
    if (
        not next_qs
        and missing_required
        and s["round"] < MAX_ROUNDS
        and out.get("collection_only")
        and not out.get("help_message")
    ):
        next_qs = [_DEMOGRAPHIC_REASK[k] for k in missing_required][:MAX_FOLLOWUP_QUESTIONS]

    s["clarifying_questions"] = next_qs
    s["asked_questions"] = list(dict.fromkeys(list(asked_questions) + list(next_qs)))
    status, reason = _conversation_status(s["round"], next_qs, out, required_missing=bool(missing_required))
    s["status"] = status
    s["status_reason"] = reason
    if "llm_clarification" in out and isinstance(out["llm_clarification"], dict):
        out["llm_clarification"]["clarifying_questions"] = next_qs
    out["session_id"] = req.session_id
    out["conversation_status"] = status
    out["conversation_status_reason"] = reason
    out["conversation_round"] = s["round"]
    out["max_rounds"] = MAX_ROUNDS
    out["ready_for_full_pipeline"] = bool(ready or emergency or force_run)
    out["emergency_signal_detected"] = bool(emergency)
    round_results = s.get("round_results") or []
    round_results.append(_build_round_result(int(s.get("round", 0)), out))
    s["round_results"] = round_results
    s["latest_risk_output"] = {
        "risk_score": out.get("risk_score"),
        "severity": out.get("severity"),
        "confidence": out.get("confidence"),
        "triage_recommendation": out.get("triage_recommendation"),
        "possible_conditions": out.get("possible_conditions", []),
    }
    _update_session(req.session_id, s)
    out["patient_conversation"] = combined
    return out


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    deleted = _delete_session(session_id)
    return {"status": "cleared" if deleted else "not_found"}


@app.post("/extract")
def extract_phase1(req: ExtractRequest):
    r = extractor.extract(req.conversation)
    return extractor.to_dict(r)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(llm.client),
        "ner_mode": pipeline.ner_mode,
        "version": "2.1",
    }
