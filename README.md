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

```
cepialabs_health_care/
├── src/
│   ├── extraction/      # Stage 1: NER + pattern rules + negation
│   ├── ontology/        # Stage 2: SNOMED mapping
│   ├── features/        # Stage 3: Feature builder
│   ├── risk_model/      # Stage 4: Probabilistic inference
│   ├── llm_reasoning/   # Stage 5: LLM clarification
│   └── pipeline.py     # Orchestrator
├── api/
│   └── main.py          # FastAPI server
├── docs/
│   └── PIPELINE.md      # Detailed pipeline docs
└── requirements.txt
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

## Next Steps

1. **Stage 1:** Integrate Medical NER (BioBERT, ClinicalBERT, or custom model)
2. **Stage 2:** Add full SNOMED CT / UMLS ontology
3. **Stage 4:** Train risk model on symptom-disease datasets
4. **Stage 5:** Connect OpenAI GPT-4 or local LLM
5. Add PubMed research tool (per Krish Naik reference)
6. Implement MCP Server for agentic tools
