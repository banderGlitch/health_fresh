# AI-Analyzer | Medical Diagnosis Pipeline

**By CepiaLabs** — An intelligent medical research and diagnostic pipeline.

Reference: [Krish Naik - Medical Diagnosis App](https://www.krishnaik.in/project/medical-diagnosis-app)

---

## Objectives

The module must:

1. **Extract symptoms precisely** — From natural language conversation
2. **Normalize to medical ontology** — SNOMED CT, synonym expansion
3. **Capture context** — Duration, severity, triggers
4. **Infer possible conditions** — Probabilistic inference
5. **Estimate risk & uncertainty** — Risk scores, confidence

---

## Pipeline Flow

```
Conversation → NER Extraction → Ontology Mapping → Feature Builder → Risk Model → LLM Clarification → Final Symptom Profile
```

---

## Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for full layout.

```
api/         → FastAPI routes
src/         → Pipeline (extraction, ontology, features, risk, llm)
scripts/     → NER training
data/        → Training data
models/      → Trained NER model (gitignored)
docs/        → Documentation
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload
```

**Phase 1 (extraction only):**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"conversation": "I have had fever for 3 days and a mild headache. No vomiting."}'
```

**Full pipeline:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"conversation": "I have had fever for 3 days and a mild headache. No vomiting."}'
```

---

## Phase 1 – ML NER (v2.1)

Phase 1 now uses an **in-house trained DistilBERT** model for symptom extraction:

- **Model:** Fine-tuned token classification (B-SYMPTOM, I-SYMPTOM, O)
- **Data:** Synthetic training data from symptom lexicon
- **Performance:** ~99.7% F1 on validation
- **Duration/severity/negation:** Still handled by rule-based post-processing

To retrain: `python scripts/prepare_ner_data.py` then `python scripts/train_ner.py`.

---

## Next Steps

1. ~~**Stage 1:** Integrate Medical NER~~ ✅ Done (in-house DistilBERT)
2. **Stage 2:** Add full SNOMED CT / UMLS ontology
3. **Stage 4:** Train risk model on symptom-disease datasets
4. **Stage 5:** Connect OpenAI GPT-4 or local LLM
5. Add PubMed research tool (per Krish Naik reference)
6. Implement MCP Server for agentic tools
