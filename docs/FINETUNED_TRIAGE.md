# Finetuned Triage Risk Model Integration

The pipeline prefers finetuned triage (3-way) when available. Model files are in `models/finetuned_triage/`.

**Output:** OTC Drug | Doctor Consultation | **Emergency** (vs SYNAPSE's 2-way)

---

## Input Mapping

Pipeline features → finetuned triage via symptom mapping:

| Pipeline | Triage model | Notes |
|----------|--------------|-------|
| `symptom_text`, `severities`, `max_duration_days` | Symptom category | Mapped to one of 20 categories (see `triage_mapper.py`) |
| `age` | Age bucket | 16-30, 31-50, 50+, 6-15 |
| `gender` | Gender | Male, Female |
| `max_duration_days`, `acute_flag` | Duration | 1-2 days, 3-5 days, 1 week, 2+ weeks |
| `severities` | Severity | Mild, Moderate, Severe (worst used) |

---

## Output Mapping

| Triage | RiskScore | Severity |
|--------|-----------|----------|
| OTC Drug | 0.3 | LOW |
| Doctor Consultation | 0.8 | HIGH |
| **Emergency** | **0.95** | **HIGH** |

Red-flag override: `has_red_flag` + `red_flag_severe` → Emergency.

---

## Symptom Categories (20)

Breathing shortness, Chest tightness and sweating, Cold and throat irritation, Cough and cold, Fever and mild headache, Headache since one week, High blood pressure symptoms, High fever for 5 days, Low grade fever, Mild back pain, Mild fever, Mild skin rash, Persistent cough, Severe abdominal pain, Severe allergic reaction, Severe breathing difficulty, Severe chest pain, Severe migraine, Stomach pain and vomiting, Vomiting and dehydration.

---

## Decision Flow

1. **Finetuned triage** (if models loaded) → 3-way triage
2. **SYNAPSE** (if finetuned triage unavailable) → 2-way triage
3. **Heuristic** (if both fail) → severity + red flags

---

## Disable Finetuned Triage

```python
RiskPredictor(use_finetuned_triage=False, use_synapse=True)
```

---

## Files

- `src/risk_model/triage_mapper.py` — Symptom + feature mapping
- `src/risk_model/finetuned_triage_predictor.py` — Predictor adapter
- `src/risk_model/predictor.py` — Orchestrates finetuned triage → SYNAPSE → heuristic
- `models/finetuned_triage/` — risk_model.pkl, label_encoders.pkl
