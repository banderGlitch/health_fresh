# Project Structure

> AI-Analyzer — Medical Diagnosis Pipeline

---

## Folder Layout

```
cepialabs_health_care/
│
├── api/                      # REST API
│   ├── main.py               # FastAPI app, routes
│   └── __init__.py
│
├── src/                      # Core pipeline
│   ├── pipeline.py           # Orchestrator (runs all stages)
│   ├── extraction/           # Phase 1: NER
│   │   ├── ml_ner_extractor.py   # ML model (symptom spans)
│   │   ├── ner_extractor.py     # Rule-based (duration, severity, negation)
│   │   ├── symptom_lexicon.py   # Symptom vocabulary
│   │   └── __init__.py
│   ├── ontology/             # Phase 2: SNOMED mapping
│   │   ├── mapper.py
│   │   └── __init__.py
│   ├── features/             # Phase 3: Feature builder
│   │   ├── builder.py
│   │   ├── risk_features.py   # Red flags, syndromes, severity encoding
│   │   └── __init__.py
│   ├── risk_model/           # Phase 4: Risk scoring
│   │   ├── predictor.py
│   │   └── __init__.py
│   └── llm_reasoning/        # Phase 5: LLM clarification
│       ├── reasoner.py
│       └── __init__.py
│
├── scripts/                  # NER training (standalone)
│   ├── prepare_ner_data.py   # Generate training data
│   ├── train_ner.py          # Train DistilBERT
│   ├── test_ml_ner.py        # Test trained model
│   └── README_NER.md         # NER training docs
│
├── data/                     # Data
│   ├── ner/                  # CoNLL training data
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   └── Corona2.json          # Medical NER dataset (future use)
│
├── models/                   # Trained model (gitignored)
│   └── ner_symptom/          # DistilBERT NER weights
│
├── tests/
│   └── test_extraction.py    # Phase 1 tests
│
├── docs/
│   ├── PIPELINE.md
│   ├── CODE_EXPLANATION.md
│   ├── PROJECT_ROADMAP_FOR_STAKEHOLDERS.md
│   └── PROJECT_STRUCTURE.md
│
├── .env.example              # Env template (GROQ_API_KEY, etc.)
├── .gitignore
├── requirements.txt
├── requirements-ner.txt      # NER training deps
├── run_server.bat            # Start API (Windows CMD)
├── run_server.ps1            # Start API (PowerShell)
├── test_health.bat           # Quick health check
├── CHANGELOG.md
└── README.md
```

---

## Data Flow

```
Request (conversation)
    ↓
api/main.py  →  pipeline.run()
    ↓
extraction.MLNERExtractor  (symptoms, duration, severity, negation)
    ↓
ontology.OntologyMapper   (SNOMED codes)
    ↓
features.FeatureBuilder   (feature dict)
    ↓
risk_model.RiskPredictor  (risk score, severity)
    ↓
llm_reasoning.LLMReasoner (clarifying questions)
    ↓
Response (symptoms, risk, clarification)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI routes, request handling |
| `src/pipeline.py` | Pipeline orchestration |
| `src/extraction/ml_ner_extractor.py` | ML symptom extraction |
| `src/extraction/ner_extractor.py` | Rule-based (used for duration/severity/negation) |
| `src/llm_reasoning/reasoner.py` | LLM calls (Groq, Gemini, OpenAI) |
| `scripts/train_ner.py` | NER model training |
