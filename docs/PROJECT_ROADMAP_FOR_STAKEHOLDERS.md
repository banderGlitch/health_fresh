# AI-Analyzer: Project Roadmap & Status Report

**For:** Management / Mentors  
**From:** Development Team  
**Date:** February 2025  
**Project:** AI-Analyzer — Medical Diagnosis Pipeline (CepiaLabs)

---

## 1. Executive Summary

AI-Analyzer is an intelligent medical diagnosis pipeline that converts patient conversations into structured symptom profiles. The system uses a hybrid approach: **machine learning** for symptom extraction and **rule-based logic** for context (duration, severity, negation). Phase 1 (NER) has been upgraded with an in-house trained ML model, replacing the previous lexicon-based approach.

**Current Status:** Phase 1 ML integration complete. Pipeline operational end-to-end.

---

## 2. Pipeline Overview

```
Patient Conversation
        ↓
   [Phase 1] NER Extraction (ML + Rules)
        ↓
   [Phase 2] Ontology Mapping (SNOMED)
        ↓
   [Phase 3] Feature Builder
        ↓
   [Phase 4] Risk Model
        ↓
   [Phase 5] LLM Clarification
        ↓
   Final Structured Symptom Profile
```

---

## 3. Phase-wise Roadmap

### Phase 1: Structured NLP Extraction — ✅ COMPLETE (ML Integrated)

| Component | Implementation | Status |
|-----------|----------------|--------|
| Symptom spans | ML model (DistilBERT, fine-tuned) | ✅ In production |
| Duration | Rule-based (regex) | ✅ Active |
| Severity | Rule-based (mild/moderate/severe) | ✅ Active |
| Negation | Rule-based ("no X", "denies X") | ✅ Active |
| Associated factors | Rule-based | ✅ Active |

**ML Model Details:**
- Architecture: DistilBERT (token classification)
- Training data: Synthetic data from symptom lexicon (~5,400 sentences)
- Validation F1: 99.7%
- Model location: `models/ner_symptom/`

---

### Phase 2: Ontology Mapping — ✅ IMPLEMENTED

- Maps extracted symptoms to SNOMED CT codes
- Handles synonyms (e.g., "pyrexia" → Fever)
- ~10 core symptoms mapped

---

### Phase 3: Feature Builder — ✅ IMPLEMENTED

- Builds feature vectors from symptoms, demographics, history
- Input to Risk Model

---

### Phase 4: Risk Model — ⚠️ PLACEHOLDER

- Currently: rule-based (mild→LOW, severe→HIGH, else MODERATE)
- **Future:** Train XGBoost/LightGBM on symptom-disease datasets

---

### Phase 5: LLM Clarification — ✅ IMPLEMENTED

- Groq (primary) → Gemini → OpenAI fallback
- Generates reasoning summary and clarifying questions
- Guardrail: Severity comes from Risk Model, not LLM

---

## 4. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, NER mode, LLM status |
| `/extract` | POST | Phase 1 only — symptom extraction |
| `/analyze` | POST | Full pipeline |
| `/analyze/continue` | POST | Follow-up with patient answers |
| `/session/{id}` | DELETE | Clear session |
| `/docs` | GET | Swagger UI (interactive testing) |
| `/redoc` | GET | API documentation |

**Sample Request (JSON):**
```json
{
  "conversation": "I've had fever for 3 days and a mild headache. No vomiting.",
  "demographics": { "age": 45, "gender": "male" },
  "history": {}
}
```

---

## 5. Technology Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| NLP / ML | Transformers, PyTorch, DistilBERT |
| LLM | Groq, Google Gemini, OpenAI |
| Data | Synthetic training data, symptom lexicon |
| Config | Python, .env |

---

## 6. Project Structure (High-Level)

```
cepialabs_health_care/
├── api/               # FastAPI server
├── src/
│   ├── extraction/    # Phase 1: ML NER + rules
│   ├── ontology/      # Phase 2: SNOMED mapping
│   ├── features/      # Phase 3: Feature builder
│   ├── risk_model/    # Phase 4: Risk scoring
│   ├── llm_reasoning/ # Phase 5: LLM clarification
│   └── pipeline.py    # Orchestrator
├── scripts/           # NER training scripts
├── data/              # Training data, Corona2.json
├── models/            # Trained NER model
└── docs/              # Documentation
```

---

## 7. Next Steps (Recommended)

| Priority | Task | Effort |
|----------|------|--------|
| 1 | Integrate Corona2.json into NER training (real medical text) | Medium |
| 2 | Train proper Risk Model (XGBoost/LightGBM) on symptom-disease data | High |
| 3 | Add MedDialog / doctor-patient datasets for NER fine-tuning | Medium |
| 4 | Expand SNOMED ontology coverage | Medium |
| 5 | Add Redis/DB for session storage (production) | Low |
| 6 | Deploy to cloud (e.g., Azure, AWS) | Medium |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ML model trained on synthetic data only | Add real medical datasets (Corona2, MedDialog) |
| Risk Model is heuristic | Train on symptom-disease datasets |
| Session store in-memory | Use Redis/DB for production |
| API keys in .env | Use secrets manager in production |

---

## 9. Deliverables (Current)

- [x] Phase 1 ML NER model trained and integrated
- [x] FastAPI server with /extract, /analyze, /analyze/continue
- [x] Swagger UI for testing
- [x] Changelog and documentation
- [x] Rule-based fallback available

---

## 10. How to Run

```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn api.main:app --reload

# Test
curl http://localhost:8000/health
# Open http://localhost:8000/docs for interactive testing
```

---

## 11. Contact & References

- **Reference:** [Krish Naik - Medical Diagnosis App](https://www.krishnaik.in/project/medical-diagnosis-app)
- **Repository:** [GitHub - ai_health](https://github.com/banderGlitch/ai_health)

---

*Document prepared for internal stakeholder review. Last updated: February 2025.*
