# SYNAPSE Risk Model Integration

SYNAPSE is used as **fallback** when finetuned triage is unavailable. See `docs/FINETUNED_TRIAGE.md` for the primary model.

SYNAPSE: OTC Drug vs Doctor Consultation (2-way). Model files in `models/synapse_triage/`.

**Dependencies:** scikit-learn, scipy, joblib (for model loading).

---

## Input Mapping

Pipeline features → SYNAPSE model input:

| Pipeline | SYNAPSE | Encoding |
|----------|---------|----------|
| `symptom_text` | Symptoms (TF-IDF) | Comma-separated from FeatureBuilder |
| `age` (int) | Age | 0=16-45, 1=16-60, 2=6-15, 3=above 45, 4=above 60, 5=below 5 |
| `gender` | Gender | 0=Female, 1=Male |
| `max_duration_days`, `acute_flag` | Duration | 0=Greater than 3 days, 1=Less than 3 days |
| `severities` | Severity | 0=Mild, 1=Moderate, 2=Severe (worst used) |

---

## Output Mapping

| SYNAPSE | RiskScore | Severity |
|---------|-----------|----------|
| OTC Drug | 0.3 | LOW |
| Doctor Consultation | 0.8 | HIGH |

Red-flag override: `has_red_flag` + `red_flag_severe` → 0.95, HIGH.

---

## Demographics

Pass `demographics={"age": 35, "gender": "male"}` to `pipeline.run()`.
If omitted, defaults: age=40, gender="unknown" (treated as Female).

---

## Fallback

If the model fails to load, `RiskPredictor` uses the heuristic (severity + red flags + syndromes).

---

## Files

- `src/risk_model/synapse_predictor.py` — Adapter and predictor
- `src/risk_model/predictor.py` — Delegates to SYNAPSE when available
- `models/synapse_triage/` — Model, vectorizer, scaler, label_encoder
