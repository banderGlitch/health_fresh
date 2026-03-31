# AI-Analyzer | Medical Diagnosis Pipeline

**By BanderSnatch** — An intelligent medical research and diagnostic pipeline.

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
rag_triage/  → RAG triage (SYNAPSE embeddings + retrieval)
frontend/    → React UI (Vite) — symptom selection + /analyze
scripts/     → NER training
data/        → Training data
models/      → Trained models (not included in repo)
docs/        → Documentation
```

**Repository:** [github.com/banderGlitch/healthcare](https://github.com/banderGlitch/healthcare)

> **Note:** The `models/` folder is **not included** in this repository (gitignored due to size ~1.5 GB). You must obtain or train the models separately to run the full pipeline. See Phase 1 section below for NER training instructions.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API (port 8002; or use run_server.bat / run_server.ps1)
uvicorn api.main:app --host 127.0.0.1 --port 8002
```

**Phase 1 (extraction only):**
```bash
curl -X POST http://localhost:8002/extract \
  -H "Content-Type: application/json" \
  -d '{"conversation": "I have had fever for 3 days and a mild headache. No vomiting."}'
```

**Full pipeline:**
```bash
curl -X POST http://localhost:8002/analyze \
  -H "Content-Type: application/json" \
  -d '{"conversation": "I have had fever for 3 days and a mild headache. No vomiting."}'
```

---

## RAG triage (optional)

Build the vector index once (see `rag_triage/README.md`), then the risk pipeline can use **RAG** retrieval over SYNAPSE before falling back to other models.

```bash
cd rag_triage
python build_index.py
```

---

## Web frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

Opens the Vite app (proxies API calls to `http://127.0.0.1:8002` in dev). Start the API with `run_server.bat` first.

---

## Phase 1 – ML NER (v2.1)

Phase 1 now uses an **in-house trained DistilBERT** model for symptom extraction:

- **Model:** Fine-tuned token classification (B-SYMPTOM, I-SYMPTOM, O)
- **Data:** Synthetic training data from symptom lexicon
- **Performance:** ~99.7% F1 on validation
- **Duration/severity/negation:** Still handled by rule-based post-processing

To retrain: `python scripts/prepare_ner_data.py` then `python scripts/train_ner.py`.


