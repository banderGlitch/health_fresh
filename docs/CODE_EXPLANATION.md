# AI-Analyzer — Code Explanation (Simple Language)

A line-by-line walkthrough of the codebase.

---

## 1. symptom_lexicon.py — The Symptom Dictionary

```
Line 7: SYMPTOM_LEXICON = { ... }
```
A dictionary where each key is a standard symptom name, and the value is a list of ways people might say it.

Example: `"fever"` can be said as "fever", "feverish", "pyrexia", "high temperature", etc.

```
Lines 39-42: VARIATION_TO_CANONICAL
```
Another dictionary built from the first one. It maps every variation back to the standard name.

Example: "pyrexia" → "fever", "head hurts" → "headache"

```
Lines 45-47: get_canonical(symptom_text)
```
A function that takes any symptom phrase and returns the standard name, or `None` if not found.

---

## 2. ner_extractor.py — Phase 1: Extracting Medical Facts

### Data Structures (Lines 14-26)

```
ExtractedSymptom
```
A small container for one symptom with: name, duration, severity, and related symptoms.

```
ExtractionResult
```
Holds the full result: list of symptoms + list of negated symptoms.

### Patterns (Lines 40-68)

```
SEVERITY_PATTERNS
```
Regex patterns to find words like "mild", "moderate", "severe" in text.

```
DURATION_PATTERN
```
Finds phrases like "for 3 days", "since last week", "lasting 2 hours".

```
NEGATION_PHRASES
```
Patterns for phrases that mean "NOT having" something: "no fever", "without pain", "denies vomiting", etc.

### Key Methods

```
_build_symptom_patterns() (Lines 75-84)
```
Creates a regex for each symptom variation. Sorts by length (longest first) so "chest pain" is matched before "chest".

```
_extract_negations() (Lines 86-99)
```
1. Loops over each negation pattern
2. Finds matches in the text
3. Splits "X or Y" into separate symptoms (e.g., "vomiting or shortness of breath" → both negated)
4. Converts each to canonical name and adds to the negated list

```
_extract_duration_near() (Lines 102-114)
```
Looks at text around a symptom (80 chars before/after) and tries to find a duration phrase.

```
_extract_severity_near() (Lines 116-128)
```
Looks at text before a symptom (50 chars). Finds severity words and picks the one closest to the symptom (e.g., "mild" in "mild headache").

```
_get_associated_symptoms() (Lines 130-136)
```
Returns other symptoms found in the same text (e.g., fever and headache together).

```
extract() (Lines 138-185) — Main method
```
1. Cleans the text (extra spaces)
2. Extracts negations
3. Finds all symptom mentions using the lexicon patterns
4. Removes overlaps (keeps longest match)
5. Skips symptoms that are negated
6. Skips symptoms that appear right after "no", "not", "without"
7. For each symptom: gets duration, severity, associated factors
8. Returns ExtractionResult

```
to_dict() (Lines 187-201)
```
Converts the result into a plain dictionary for the API response.

---

## 3. mapper.py (Ontology) — Phase 2: Standard Medical Codes

```
Lines 10-21: SNOMED_SYMPTOM_MAP
```
A dictionary mapping symptom names to (code, official_name). Example: "fever" → ("386661006", "Fever").

```
map_symptom() (Lines 29-39)
```
Takes a symptom name, looks it up in the map, and returns a dict with snomed_code, canonical_name, original.

```
map_extraction_result() (Lines 41-51)
```
Takes the full extraction output, maps each symptom to SNOMED, and adds duration/severity from the extraction.

---

## 4. builder.py (Features) — Phase 3: Preparing Data for Risk Model

```
build() (Lines 17-38)
```
1. Takes mapped symptoms, demographics, and history
2. Collects SNOMED codes from symptoms
3. Returns a dict with: symptom_codes, symptom_vector, demographics, history, severities, durations

This is the input format for the risk model (when it's trained).

---

## 5. predictor.py (Risk Model) — Phase 4: Risk Assessment

```
predict() (Lines 19-41)
```
Right now it's a placeholder:
- Default: severity = MODERATE, risk_score = 0.5
- If first symptom is "severe" → severity = HIGH, risk_score = 0.85
- If first symptom is "mild" → severity = LOW, risk_score = 0.25
- Returns RiskScore, Severity, Confidence, possible_conditions

In production, this would use a trained model (e.g., XGBoost).

---

## 6. reasoner.py (LLM) — Phase 5: LLM Clarification

```
clarify() (Lines 19-33)
```
Placeholder for LLM integration. Takes conversation, extraction result, and risk output. Returns a dict with clarifying_questions, reasoning_summary, and risk_output.

Severity is never changed by the LLM — it comes from the risk engine, as per the guardrail.

---

## 7. pipeline.py — The Main Orchestrator

```
__init__() (Lines 21-26)
```
Creates one instance of each stage: extractor, ontology, feature_builder, risk_predictor, llm_reasoner.

```
run() (Lines 28-63)
```
Runs the full pipeline in order:

1. **extract** — Phase 1: get symptoms from text
2. **to_dict** — convert to dict
3. **map_extraction_result** — Phase 2: add SNOMED codes
4. **build** — Phase 3: prepare features
5. **predict** — Phase 4: get risk score
6. **clarify** — Phase 5: LLM (placeholder)
7. **return** — combine everything into the final response

---

## 8. api/main.py — The Web Server

```
Lines 10-12: sys.path
```
Adds the project root to Python path so imports work.

```
Lines 17-19: app, pipeline, extractor
```
Creates the FastAPI app and the pipeline/extractor instances.

```
Lines 22-29: Request models
```
- **AnalyzeRequest** — conversation + optional demographics + history
- **ExtractRequest** — just conversation

```
POST /analyze (Lines 32-39)
```
Runs the full pipeline and returns the complete result.

```
POST /extract (Lines 42-46)
```
Runs only Phase 1 (extraction) and returns symptoms + negated.

```
GET /health (Lines 49-51)
```
Returns `{"status": "ok"}` to check if the server is running.

---

## Flow Summary

```
User sends: "I have fever for 3 days. No vomiting."

1. POST /analyze receives it
2. pipeline.run() is called
3. extractor.extract() → finds fever, 3 days, negated: vomiting
4. ontology.map_extraction_result() → maps fever to SNOMED 386661006
5. feature_builder.build() → prepares symptom codes, etc.
6. risk_predictor.predict() → returns risk_score, severity
7. llm_reasoner.clarify() → placeholder, passes through
8. Response is sent back to user
```
