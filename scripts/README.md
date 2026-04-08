# Scripts

| Script | Purpose |
|--------|---------|
| `prepare_ner_data.py` | Generate synthetic NER training data from symptom lexicon |
| `train_ner.py` | Fine-tune DistilBERT for symptom NER |
| `test_ml_ner.py` | Test ML NER extractor on sample sentences |
| `test_finetuned_triage.py` | Test finetuned triage model (OTC\|Doctor\|Emergency) |
| `test_clarification_flow.py` | Test POST /analyze → POST /analyze/continue flow |
| `compare_synapse_vs_finetuned.py` | Compare SYNAPSE (2-way) vs finetuned triage (3-way) |

See `README_NER.md` for NER training setup.
