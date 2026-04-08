# Feature Builder Output — Quick Reference

Output from `FeatureBuilder.build()` (lines 99–128 in `builder.py`). These features feed the **Risk Model** (finetuned triage, SYNAPSE, heuristic).

---

## Feature Table

| Feature | Type | Description | Used By |
|---------|------|--------------|---------|
| **Legacy / raw** |
| `symptom_codes` | list | SNOMED codes from ontology | — |
| `symptom_vector` | list | Unique symptom codes | — |
| `symptom_text` | str | Comma-separated symptom names | SYNAPSE (TF-IDF), triage_mapper |
| `demographics` | dict | Raw age, gender from request | — |
| `history` | dict | Raw medical history | — |
| `severities` | list | Per-symptom severity strings | triage_mapper |
| `durations` | list | Per-symptom duration strings | — |
| **Symptom counts** |
| `symptom_count` | int | Number of symptoms | heuristic |
| **Severity** |
| `max_severity` | float | Worst severity (0.33–1.0) | heuristic |
| `has_severe` | 0/1 | Any severe symptom | heuristic, triage_mapper |
| `severe_count` | int | Count of severe symptoms | heuristic |
| `severity_weighted_sum` | float | Sum of severity weights | heuristic |
| **Duration** |
| `max_duration_days` | float | Longest duration in days | triage_mapper, heuristic |
| `acute_flag` | 0/1 | Onset &lt; 24h | triage_mapper, heuristic |
| `chronic_flag` | 0/1 | Duration &gt; 2 weeks | heuristic |
| **Red flags** |
| `has_red_flag` | 0/1 | Chest pain, dyspnea, blood, etc. | heuristic, finetuned |
| `red_flag_count` | int | Number of red-flag symptoms | heuristic |
| `red_flag_severe` | 0/1 | Red-flag + severe | heuristic, finetuned |
| **Syndromes** |
| `syndrome_cardiac_like` | 0/1 | Chest pain + shortness of breath | heuristic, possible_conditions |
| `syndrome_respiratory` | 0/1 | Fever + cough + fatigue | heuristic, possible_conditions |
| `syndrome_gi` | 0/1 | Abdominal pain + vomiting/diarrhea | heuristic, possible_conditions |
| `syndrome_alarm` | 0/1 | Coughing blood, blood in stool | heuristic |
| **Demographics** |
| `age` | int | Patient age (default 40) | triage_mapper, heuristic |
| `age_group` | 0–3 | &lt;18, 18–40, 40–60, 60+ | heuristic |
| `gender` | str | male/female/unknown | triage_mapper |
| **Comorbidities** |
| `has_diabetes` | 0/1 | From history | heuristic |
| `has_hypertension` | 0/1 | From history | heuristic |
| `has_heart_disease` | 0/1 | From history | heuristic |
| `has_lung_disease` | 0/1 | From history | heuristic |
| `comorbidity_count` | int | Sum of above | heuristic |

---

## Consumer Summary

| Consumer | Key Features Used |
|----------|-------------------|
| **Finetuned triage** | `symptom_text`, `severities`, `max_duration_days`, `acute_flag`, `age`, `gender` → via `triage_mapper` |
| **SYNAPSE** | `symptom_text` (TF-IDF), `age`, `gender`, `max_duration_days`, `acute_flag`, `severities` |
| **Heuristic fallback** | `has_severe`, `has_red_flag`, `red_flag_severe`, `syndrome_*`, `age_group`, `comorbidity_count` |

---

## Data Flow

```
NER extraction → Ontology mapping → FeatureBuilder.build()
                                        ↓
                              features dict (above)
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
            FinetunedTriage    SynapseTriage         Heuristic
            (triage_mapper)    (vectorizer)          (_heuristic_predict)
```
