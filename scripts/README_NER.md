# In-House NER Model (Phase 1)

Train and test a symptom NER model locally—no LLM required.

## Setup

```bash
pip install -r requirements-ner.txt
```

## 1. Prepare Training Data

Generates synthetic sentences from the symptom lexicon with BIO labels.

```bash
python scripts/prepare_ner_data.py
```

Output: `data/ner/train.txt`, `val.txt`, `test.txt`, `labels.txt`

## 2. Train Model

Fine-tunes DistilBERT for token classification (B-SYMPTOM, I-SYMPTOM, O).

```bash
python scripts/train_ner.py
```

Output: `models/ner_symptom/` (config, tokenizer, model weights)

## 3. Test Model

Standalone test on sample sentences.

```bash
python scripts/test_ml_ner.py
```

## Use in Pipeline (Optional)

```python
from src.extraction.ml_ner_extractor import MLNERExtractor

extractor = MLNERExtractor()  # Loads from models/ner_symptom/
result = extractor.extract("I have fever and headache for 3 days.")
print(extractor.to_dict(result))
```
