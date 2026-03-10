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

_VALID_SEVERITIES = {"mild", "moderate", "severe"}
_NOISE_WORDS = {
    "a",
    "an",
    "and",
    "or",
    "on",
    "off",
    "of",
    "the",
    "to",
    "for",
    "with",
    "stairs",
}

# It is the model which detects the symptoms 

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
        #    "high temperature" → "fever"
            name = get_canonical(cleaned)

            if not name and cleaned in SYMPTOM_LEXICON:
                name = cleaned
            # if name itself is the canonical name  ex :- fever 
            if not name:
                # if name is not the canonical name , then check if it is a variation of the canonical name 
                for canonical, variations in SYMPTOM_LEXICON.items():
                    if cleaned in canonical or canonical in cleaned:
                        name = canonical
                        break
                   # similarly doing
                    if any(v in cleaned or cleaned in v for v in variations):
                        name = canonical
                        break
            if not name or name in seen_names:
                continue

            seen_names.add(name)  # to prevent duplicate 
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
        # " I   have  Chest   Pain  "
        # after normalizing 
        # "I have chest pain"
        normalized_text = normalize_text(text)
        if not normalized_text:
            return ExtractionResult(symptoms=[], negated=[])
        
#         symptoms = []
#         negated = []

        try:
            ner = self.pipeline(normalized_text)
        except Exception:
            return ExtractionResult(symptoms=[], negated=[])
        
#         [
#        {"word":"chest pain","start":12,"end":22},
#         {"word":"fever","start":35,"end":40}
#        ]

        if not ner:
            return ExtractionResult(symptoms=[], negated=[])


        spans = self._build_spans(normalized_text, ner)
#         spans = [
#           ("chest pain", 7, 17),
#          ("fever", 22, 27)
#    ]
        if not spans:
            return ExtractionResult(symptoms=[], negated=[])

        try:
            llm_result = self._enrich_with_llm(normalized_text, spans)
            if llm_result is not None:
                return llm_result
        except Exception:
            pass
        # if llm fails , then rule based extraction
        ## rule based extraction
        negated = self._rules.extract_negations(normalized_text)
        symptoms = self._rules.build_symptoms_from_spans(normalized_text, spans, negated)
        return ExtractionResult(symptoms=symptoms, negated=negated)

    def to_dict(self, result: ExtractionResult) -> dict[str, Any]:
        return to_result_dict(result)
