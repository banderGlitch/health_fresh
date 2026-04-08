"""
Phase 1 (ML + LLM enrichment): Symptom spans from DistilBERT;
negation/severity/duration via LLM with rule fallback.
"""

import json
import re
from pathlib import Path
from typing import Any

from ..llm_reasoning import LLMReasoner
from .ner_extractor import ExtractedSymptom, ExtractionResult, NERExtractor, normalize_text, to_result_dict
from .symptom_lexicon import SYMPTOM_LEXICON, get_canonical
from .symptom_rag import rag_spans

_VALID_SEVERITIES = {"mild", "moderate", "severe"}
_NOISE_WORDS = {
    "a", "an", "and", "or", "on", "off", "of", "the", "to", "for", "with", "stairs",
}

# No canned help_message here — avoids repeating generic "limited symptoms" copy.
# Collection-phase LLM (reasoner.classify_extraction_gap) can still guide the user when needed.

class MLNERExtractor:
    """ML finds symptom spans; LLM enriches negation/severity/duration."""

    def __init__(self, model_path: str | Path | None = None):
        root = Path(__file__).resolve().parent.parent.parent
        self.model_path = Path(model_path or root / "models" / "ner_symptom")
        self._pipeline = None
        self._rules = NERExtractor()
        self._llm = LLMReasoner()

    @property
    def pipeline(self):
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}. Run: python scripts/train_ner.py")
            from transformers import pipeline
            self._pipeline = pipeline("ner", model=str(self.model_path), aggregation_strategy="simple")
        return self._pipeline

    def _call_llm(self, prompt: str) -> str | None:
        if not self._llm.client:
            return None
        if self._llm._groq_client:
            return self._llm._call_groq(prompt)
        return None

        # remove markdown
        # parse json 
        # return dictionary 

    def _parse_llm_json(self, content: str | None) -> dict[str, Any]:
        if not content:
            return {}
        payload = content.strip()
        if payload.startswith("```"):
            payload = payload.split("```", 2)[1]
            if payload.startswith("json"):
                payload = payload[4:]
        try:
            parsed = json.loads(payload.strip())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _rephrase_for_extraction(self, text: str) -> str:
        """LLM rephrase: clarify vague terms into standard symptom phrases. Pass through original if unsure."""
        if not self._llm.client:
            return text
        prompt = """Rephrase this patient symptom description into a clear clinical sentence.

STRICT RULES:
- Use standard symptom terms: "stomach pain"/"abdominal pain" instead of "discomfort", "stomach ache" instead of "ache".
- Do NOT add symptoms the patient did not mention.
- Do NOT remove any symptom the patient mentioned.
- Do NOT diagnose or suggest conditions.
- Keep it brief. Same meaning, clearer wording only.
- If the text is already clear, ambiguous, or you are unsure, output EXACTLY: SAME

Patient text: """
        try:
            out = self._call_llm(prompt + f'"{text}"')
            if not out or not isinstance(out, str):
                return text
            out = out.strip()
            if not out or out.upper() == "SAME":
                return text
            if len(out) > len(text) * 3:
                return text
            if abs(len(out) - len(text)) > max(100, len(text) * 2):
                return text
            return out
        except Exception:
            return text

    def _phrase_search_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Step 4: Scan text for known symptom phrases; add any not from NER."""
        text_lower = text.lower()
        found: list[tuple[str, int, int]] = []
        for canonical, variations in SYMPTOM_LEXICON.items():
            for phrase in [canonical] + variations:
                idx = text_lower.find(phrase)
                if idx >= 0:
                    found.append((canonical, idx, idx + len(phrase)))
                    break  # one match per canonical
        return found

    def _filter_symptoms_by_text(self, symptoms: list[ExtractedSymptom], text: str) -> list[ExtractedSymptom]:
        """Step 3: Remove symptoms that don't appear in the original text."""
        text_lower = text.lower()
        keep: list[ExtractedSymptom] = []
        for s in symptoms:
            variations = SYMPTOM_LEXICON.get(s.name, [s.name])
            if any(phrase in text_lower for phrase in [s.name] + variations):
                keep.append(s)
        return keep

    def _build_spans(self, text: str, ner: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
        # [
        #  {"word":"chest pain","start":7,"end":17}  ,
        #  ]

        # [
        #  ("chest pain", 7, 17)
        # ]

        spans: list[tuple[str, int, int]] = []
        seen_names: set[str] = set()
        for item in ner:
            token = item.get("word", "").strip()
            if not token:
                continue
            cleaned = re.sub(r"\s+", " ", token.replace("##", "").strip().lower())
            cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)

            if len(cleaned) < 3 or cleaned in _NOISE_WORDS:
                continue

            name = get_canonical(cleaned)
            if not name and cleaned in SYMPTOM_LEXICON:
                name = cleaned
            if not name:
                for canonical, variations in SYMPTOM_LEXICON.items():
                    if cleaned == canonical or cleaned in variations:
                        name = canonical
                        break
            if not name or name in seen_names:
                continue

            seen_names.add(name)
            spans.append((name, item.get("start", 0), item.get("end", len(text))))
        return spans
# llm Helps us to detect more feature given the features extracted from the model 

# ❌ negation → "no fever"

# ⏱ duration → "for 2 days"

# 🌡 severity → "severe pain"



# "I have severe chest pain for 2 days but no fever"

# [
#  ("chest pain", 12, 22),
#  ("fever", 32, 37)
# ]



    def _enrich_with_llm(self, text: str, spans: list[tuple[str, int, int]]) -> ExtractionResult | None:
        entities = [
            {"index": i, "name": name, "mention": text[start:end], "start": start, "end": end}
            for i, (name, start, end) in enumerate(spans)
        ]

#         [
#           {"index":0,"name":"chest pain","mention":"chest pain","start":12,"end":22},
#           {"index":1,"name":"fever","mention":"fever","start":32,"end":37}
#         ]

        prompt = f"""Extract attributes for pre-detected symptom entities.
                     Return JSON only with keys: entities, negated.
                    For each entity return: index, negated (bool), severity (mild|moderate|severe|null), duration (string|null).

                Patient text: "{text}"
                Entities: {json.dumps(entities, ensure_ascii=True)}"""
        data = self._parse_llm_json(self._call_llm(prompt))

#         {
#        "entities":[
#               {"index":0,"negated":false,"severity":"severe","duration":"2 days"},
#               {"index":1,"negated":true,"severity":null,"duration":null}
#              ],
#        "negated":["fever"]
#       }

        rows = data.get("entities", [])

        if not isinstance(rows, list):
            return None

        by_index = {r.get("index"): r for r in rows if isinstance(r, dict) and isinstance(r.get("index"), int)}

#        rows =  [
#              {"index":0,...},
#              {"index":1,...}
#            ]

#         by doing byindex 

#        [
#          {"0,...},
#          {"1,...}
#        ]

        neg_set = {
            n.strip().lower()
            for n in data.get("negated", [])
            if isinstance(n, str) and n.strip()
        } 
        #{ negated : ["fever"] }

        symptoms: list[ExtractedSymptom] = []
        negated_names: set[str] = set()
        for i, (name, start, end) in enumerate(spans):
            row = by_index.get(i, {})
            is_negated = bool(row.get("negated")) or name.lower() in neg_set
            if is_negated:
                negated_names.add(name)
                continue

            duration = row.get("duration")
            ## if duration is not present in the row , then extract it from the text
            if not isinstance(duration, str) or not duration.strip():
                duration = self._rules._extract_duration_near(text, start, end)
            else:
                duration = duration.strip()

            severity = row.get("severity")
            # if severity is not present in the row , then extract it from the text
            if severity not in _VALID_SEVERITIES:
                severity = self._rules._extract_severity_near(text, start, end)

            symptoms.append(ExtractedSymptom(name=name, duration=duration, severity=severity))

        names = [s.name for s in symptoms]
        # ["chest pain", "shortness of breath", "sweating"]


        for s in symptoms:
            s.associated_factors = [n for n in names if n != s.name]
#             chest pain             
#             associated_factors = [
#                        "shortness of breath",
#                         "sweating"
#                 ]


        return ExtractionResult(symptoms=symptoms, negated=sorted(negated_names))

        
    #  ExtractedSymptom(
    # name="chest pain",
    # duration="2 days",
    # severity="severe",
    # associated_factors=["shortness of breath","sweating"]
# )

    def extract(self, text: str) -> ExtractionResult:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return ExtractionResult(symptoms=[], negated=[])

        text_for_extraction = self._rephrase_for_extraction(normalized_text)

        try:
            ner = self.pipeline(text_for_extraction)
        except Exception:
            return ExtractionResult(symptoms=[], negated=[], help_message=None)

        # Build spans from NER (or empty if NER returned nothing)
        spans = self._build_spans(text_for_extraction, ner) if ner else []
        # Always run phrase search - catches fragmented input like "Fever. Headache. 3 days"
        phrase_spans = self._phrase_search_spans(text_for_extraction)
        seen_names = {s[0] for s in spans}
        for name, start, end in phrase_spans:
            if name not in seen_names:
                spans.append((name, start, end))
                seen_names.add(name)
        try:
            for name, start, end in rag_spans(text_for_extraction):
                if name not in seen_names:
                    spans.append((name, start, end))
                    seen_names.add(name)
        except Exception:
            pass

        if not spans:
            return ExtractionResult(symptoms=[], negated=[], help_message=None)

        try:
            llm_result = self._enrich_with_llm(text_for_extraction, spans)
            if llm_result is not None:
                symptoms = self._filter_symptoms_by_text(llm_result.symptoms, text_for_extraction)
                if symptoms:
                    return ExtractionResult(
                        symptoms=symptoms, negated=llm_result.negated, help_message=None
                    )
                # LLM dropped all entities — fall through to rule-based build from spans
        except Exception:
            pass

        negated = self._rules.extract_negations(text_for_extraction)
        symptoms = self._rules.build_symptoms_from_spans(text_for_extraction, spans, negated)
        symptoms = self._filter_symptoms_by_text(symptoms, text_for_extraction)
        return ExtractionResult(symptoms=symptoms, negated=negated, help_message=None)

    def to_dict(self, result: ExtractionResult) -> dict[str, Any]:
        return to_result_dict(result)
