"""
Stage 5: LLM Clinical Reasoning
Handles vague descriptions, multi-symptom reasoning, clarifying questions.
IMPORTANT: LLM cannot finalize severity - must pass through risk engine.
Supports: Groq (free), Gemini, OpenAI.
"""

import json
import logging

logger = logging.getLogger(__name__)
import os
from pathlib import Path
from typing import Any


def _load_env():
    """Ensure .env is loaded."""
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parent.parent.parent
        load_dotenv(project_root / ".env")
    except ImportError:
        pass


class LLMReasoner:
    """
    Uses LLM for clinical reasoning.
    Priority: Groq (free) -> Gemini -> OpenAI.
    Guardrail: Severity comes from risk model, not LLM.
    """

    def __init__(self, api_key: str | None = None):
        _load_env()
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self._groq_client = None
        self._gemini_client = None
        self._openai_client = None

        if self.groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_key)
            except Exception:
                pass
        if not self._groq_client and self.gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_key)
            except Exception:
                pass
        if not self._groq_client and not self._gemini_client and self.openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_key)
            except Exception:
                pass

    @property
    def client(self):
        """True if any LLM client is available."""
        return (
            self._groq_client is not None
            or self._gemini_client is not None
            or self._openai_client is not None
        )

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API. Tries multiple model names for compatibility."""
        models_to_try = (
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
        )
        last_error = None
        for model in models_to_try:
            try:
                response = self._gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": 0.3},
                )
                return (response.text or str(response)).strip()
            except Exception as e:
                last_error = e
                if "404" in str(e) or "not found" in str(e).lower():
                    continue
                raise
        raise last_error or RuntimeError("No Gemini model available")

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (free tier, fast)."""
        response = self._groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def clarify(
        self,
        conversation: str,
        extraction_result: dict[str, Any],
        risk_output: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add LLM-based clarification and reasoning.
        Uses Gemini first, then OpenAI if available.
        """
        if not self.client:
            return {
                "clarifying_questions": [],
                "reasoning_summary": "",
                "risk_output": risk_output,
            }

        symptoms = extraction_result.get("symptoms", [])
        negated = extraction_result.get("negated", [])

        prompt = f"""You are a clinical assistant. Based on this patient conversation and extracted data, provide:
1. A brief reasoning summary (2-3 sentences) about what the symptoms might suggest.
2. Clarifying questions: ask 1 or 2 only. Prefer 1 if one question covers the gap. NEVER ask 3.

RULES (strict):
- Ask only what directly helps triage: duration, severity. One short question per gap.
- NEVER ask: travel, exposure to sick people, appetite changes, vomiting/diarrhea/urination as a list.
- Keep each question short: "How long?" "How severe?" "Constant or comes and goes?"
- If enough info exists, return empty array [].

IMPORTANT: Do NOT suggest or change severity. Severity is determined by the risk engine.

Patient conversation: "{conversation}"

Extracted symptoms: {symptoms}
Negated (patient does NOT have): {negated}

Risk assessment (from engine - do not modify): Severity={risk_output.get('Severity')}, RiskScore={risk_output.get('RiskScore')}

Respond in this exact JSON format only:
{{"reasoning_summary": "your 2-3 sentence summary", "clarifying_questions": ["question 1"]}}"""

        try:
            if self._groq_client:
                content = self._call_groq(prompt)
            elif self._gemini_client:
                content = self._call_gemini(prompt)
            else:
                content = self._call_openai(prompt)

            # Parse JSON (handle markdown code blocks)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            data = json.loads(content)

            return {
                "clarifying_questions": data.get("clarifying_questions", []),
                "reasoning_summary": data.get("reasoning_summary", ""),
                "risk_output": risk_output,
            }
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                err_msg = "Rate limit or quota exceeded. Add credits or wait and retry."
            else:
                err_msg = f"LLM unavailable: {err_msg}"
            return {
                "clarifying_questions": [],
                "reasoning_summary": err_msg,
                "risk_output": risk_output,
            }

    def classify_extraction_gap(
        self,
        conversation: str,
        extraction_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Classify why extraction failed or is incomplete. Returns scenario and appropriate response.
        - vague: User gave vague description (e.g. "not feeling well") -> clarifying_questions only
        - unrecognized: User used terms we don't recognize (e.g. "diaphoresis") -> help_message only
        - partial: Some symptoms found, some not -> help_message for unrecognized terms only
        - normal: All good -> clarifying_questions for duration/severity
        """
        if not self.client:
            return {
                "scenario": "normal",
                "help_message": None,
                "clarifying_questions": [],
                "unrecognized_terms": [],
            }

        symptoms = extraction_dict.get("symptoms") or []
        extracted_names = [str(s.get("name", "")).strip() for s in symptoms if s.get("name")]
        symptom_count = len(extracted_names)

        prompt = f"""You are a clinical triage assistant. The user described their symptoms. We extracted: {extracted_names}.

Classify the situation:

1. VAGUE: User gave a vague/generic description with no specific symptoms (e.g. "I am not feeling well", "feel bad", "something wrong"). No medical terms.
2. UNRECOGNIZED: User used medical/symptom terms we don't recognize (e.g. "diaphoresis", "pyrexia"). We extracted 0 symptoms.
3. PARTIAL: We extracted some symptoms but the user mentioned more that we couldn't recognize. List the unrecognized terms.
4. NORMAL: We extracted symptoms. No unrecognized terms. Ask for duration/severity.

User said: "{conversation}"
Extracted: {extracted_names}

Respond in this exact JSON format only:
{{"scenario": "vague"|"unrecognized"|"partial"|"normal", "unrecognized_terms": ["term1", "term2"], "help_message": "message or null", "clarifying_questions": ["q1", "q2"]}}

RULES:
- If scenario is "vague": help_message=null, clarifying_questions=["Can you be more specific? What symptoms are you experiencing?"] or similar
- If scenario is "unrecognized" or "partial": help_message="We couldn't recognize [terms]. Could you describe your symptoms again using simpler words (e.g. sweating, fever, headache)?", clarifying_questions=[]
- If scenario is "normal": help_message=null, clarifying_questions=1-2 short questions about duration or severity
- unrecognized_terms: list only when scenario is "unrecognized" or "partial"
"""

        try:
            if self._groq_client:
                content = self._call_groq(prompt)
            elif self._gemini_client:
                content = self._call_gemini(prompt)
            else:
                content = self._call_openai(prompt)

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            data = json.loads(content)

            scenario = str(data.get("scenario", "normal")).lower()
            if scenario not in ("vague", "unrecognized", "partial", "normal"):
                scenario = "normal"

            return {
                "scenario": scenario,
                "help_message": data.get("help_message"),
                "clarifying_questions": data.get("clarifying_questions") or [],
                "unrecognized_terms": data.get("unrecognized_terms") or [],
            }
        except Exception as e:
            logger.warning("[CLASSIFY] LLM failed: %s", e)
            return {
                "scenario": "normal",
                "help_message": None,
                "clarifying_questions": [],
                "unrecognized_terms": [],
            }

    def merge_clarification(
        self,
        conversation: str,
        clarifying_questions: list[str],
        patient_answers: str,
    ) -> str:
        """
        Merge original conversation + Q&A into a clear clinical narrative.
        Ensures answers (e.g. "3 days", "mild") are correctly linked to symptoms
        so NER can extract structured data.
        """
        logger.info("[CLARIFICATION] Merge requested: original=%d chars, %d questions, answers=%d chars",
                    len(conversation), len(clarifying_questions), len(patient_answers))

        if not self.client:
            logger.info("[CLARIFICATION] LLM unavailable, using simple append")
            return f"{conversation}\n\nPatient clarification: {patient_answers}"

        q_list = "\n".join(f"- {q}" for q in clarifying_questions) if clarifying_questions else "(no specific questions)"

        prompt = f"""You are a clinical scribe. Merge the patient's initial complaint and their answers into ONE clear clinical narrative.

STRICT RULES:
1. Link each answer to the right question: "3 days" → duration for symptoms, "mild"/"moderate"/"severe" → severity.
2. Phrase explicitly: "fever and headache for 3 days", "mild severity", "runny nose for a week".
3. Use exact words NER expects: symptom names (fever, headache, cough), duration ("3 days", "a week", "2 weeks"), severity ("mild", "moderate", "severe").
4. One coherent paragraph. No bullets, no JSON, no "Summary:" or "The patient...". Just the narrative.
5. Do NOT add diagnoses, suggestions, or questions. Only the merged symptom description.

Example output: "Patient has fever and mild headache for 3 days. Cough and runny nose for about a week. No other symptoms."

---
Original conversation:
{conversation}

---
Questions asked:
{q_list}

---
Patient answers:
{patient_answers}

---
Merged narrative:"""

        logger.info("[CLARIFICATION] Merge prompt (%d chars):\n%s", len(prompt), prompt)

        try:
            if self._groq_client:
                content = self._call_groq(prompt)
            elif self._gemini_client:
                content = self._call_gemini(prompt)
            else:
                content = self._call_openai(prompt)
            merged = content.strip()
            # Remove quotes if LLM wrapped in them
            if merged.startswith('"') and merged.endswith('"'):
                merged = merged[1:-1]
            result = merged if merged else f"{conversation}\n\nPatient clarification: {patient_answers}"
            logger.info("[CLARIFICATION] LLM merged output (%d chars): %s", len(result), result[:200] + ("..." if len(result) > 200 else ""))
            return result
        except Exception as e:
            logger.warning("[CLARIFICATION] LLM merge failed: %s, using simple append", e)
            return f"{conversation}\n\nPatient clarification: {patient_answers}"

