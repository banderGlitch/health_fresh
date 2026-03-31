# RAG Triage — Phase 1

Separate RAG-based triage module. Replaces the traditional ML risk model with retrieval-augmented triage using the SYNAPSE dataset.

**Phase 1 scope:**
- Load SYNAPSE dataset
- Embed symptom + demographic text
- Build vector index
- Retrieve top-k similar cases
- Predict triage via majority vote from retrieved cases

## Setup

```bash
cd rag_triage
pip install -r requirements.txt
```

## Build Index (first run)

```bash
python scripts/build_index.py
```

## Usage

```python
from rag_triage import RAGTriagePredictor

predictor = RAGTriagePredictor()
result = predictor.predict(
    symptom_text="fever, headache, cough",
    age=35,
    gender="male",
    duration_days=2,
    severity="moderate",
)
# result: risk_score, severity, triage_recommendation, similar_cases, confidence
```

## Data Path

By default, loads SYNAPSE CSV from `../data/SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv` (relative to this folder). Override via `SYNAPSE_DATA_PATH` in config or env.
