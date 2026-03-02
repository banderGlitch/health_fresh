# Changelog

## [2.2.0] - 2025-02-20

### Added – Finetuned Triage Risk Model Integration

**Pipeline now uses finetuned triage (3-way) when available, with SYNAPSE as fallback.**

#### What Changed

- **Finetuned Triage Predictor** – Fine-tuned SYNAPSE-style model with **3-way output**: OTC Drug | Doctor Consultation | Emergency.
- **Decision Flow** – 1) finetuned triage (if loaded) → 2) SYNAPSE → 3) heuristic fallback.
- **Symptom Mapping** – Rule-based mapping from NER output to 20 symptom categories.

#### New Files

- `src/risk_model/triage_mapper.py` – Maps pipeline features to triage input
- `src/risk_model/finetuned_triage_predictor.py` – FinetunedTriagePredictor adapter

#### Model Location

- `models/finetuned_triage/` – `risk_model.pkl`, `label_encoders.pkl`

#### Fallback

If finetuned triage models are missing, the pipeline uses SYNAPSE (2-way) or heuristic. Use `RiskPredictor(use_finetuned_triage=False)` to disable.

---

## [2.1.0] - 2025-02-23

### Added – ML NER Model Integration

**Phase 1 (NER) now uses an in-house trained machine learning model for symptom extraction.**

#### What Changed

- **ML NER Model** – Replaced rule-based (lexicon + regex) symptom detection with a fine-tuned **DistilBERT** token classification model.
- **Model Location** – Trained model saved at `models/ner_symptom/`.
- **Training Data** – Synthetic data generated from the symptom lexicon (~5,400 train, 675 val, 675 test sentences).
- **Labels** – BIO format: `O`, `B-SYMPTOM`, `I-SYMPTOM`.
- **Validation Metrics** – F1: 99.7%, Precision: 100%, Recall: 99.4%.

#### Hybrid Approach

| Component        | Method        | Notes                                           |
|-----------------|---------------|-------------------------------------------------|
| Symptom spans   | **ML model**  | Trained DistilBERT for token classification     |
| Duration        | Rule-based    | Regex patterns (e.g., "3 days", "a week")       |
| Severity        | Rule-based    | mild / moderate / severe detection              |
| Negation        | Rule-based    | "no X", "denies X", etc.                        |
| Associated factors | Rule-based | Other symptoms in same sentence                 |

#### New Files

- `scripts/prepare_ner_data.py` – Generate training data from lexicon
- `scripts/train_ner.py` – Train DistilBERT for NER
- `scripts/test_ml_ner.py` – Standalone test script
- `scripts/README_NER.md` – NER training documentation
- `src/extraction/ml_ner_extractor.py` – ML extractor class
- `requirements-ner.txt` – Dependencies for NER training

#### API

- `GET /health` – Now includes `ner_mode: "ml"`.

#### Fallback

Rule-based `NERExtractor` remains in the codebase. To revert, uncomment `NERExtractor` and comment out `MLNERExtractor` in `src/pipeline.py` and `api/main.py`.

---

## [2.0.0] – Previous

- Full pipeline: NER → Ontology → Features → Risk Model → LLM
- Rule-based Phase 1 extraction
- FastAPI endpoints: `/analyze`, `/extract`, `/analyze/continue`
