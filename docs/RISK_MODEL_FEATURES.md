# Risk Model — Feature Specification

> Phase 4: Features for predicting clinical risk, severity, and possible conditions.

---

## 1. Feature Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Symptom presence** | Which symptoms exist | fever, chest pain, cough |
| **Symptom severity** | How severe each symptom | mild, moderate, severe |
| **Duration** | How long symptoms last | 3 days, 2 weeks |
| **Red-flag symptoms** | High-risk indicators | chest pain, shortness of breath, coughing blood |
| **Symptom combinations** | Syndrome patterns | fever + cough + shortness of breath |
| **Demographics** | Age, gender | age=45, gender=male |
| **Medical history** | Comorbidities, risk factors | diabetes, hypertension |
| **Counts** | Number of symptoms, severe count | num_symptoms=5, num_severe=2 |

---

## 2. Symptom-Based Features

### 2.1 Binary presence (one-hot per symptom)

| Feature | Type | Description |
|---------|------|-------------|
| `has_fever` | binary | 1 if fever present |
| `has_chest_pain` | binary | 1 if chest pain present |
| `has_shortness_of_breath` | binary | 1 if dyspnea present |
| `has_cough` | binary | 1 if cough present |
| ... | | One per SNOMED symptom in ontology |

**Use:** Input to ML model (XGBoost, neural net). Model learns which symptoms correlate with high risk.

---

### 2.2 Symptom vector (SNOMED codes)

| Feature | Type | Description |
|---------|------|-------------|
| `symptom_codes` | list[str] | SNOMED codes present (e.g. 386661006, 29857009) |
| `symptom_count` | int | Number of distinct symptoms |

**Use:** Embedding or multi-hot encoding for models that support categorical inputs.

---

## 3. Severity Features

### 3.1 Per-symptom severity (encoded)

| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| `max_severity` | categorical | 0=mild, 1=moderate, 2=severe | Worst severity among all symptoms |
| `has_severe` | binary | 0/1 | Any symptom rated severe |
| `has_moderate` | binary | 0/1 | Any symptom rated moderate |
| `severe_count` | int | 0,1,2,... | Number of severe symptoms |
| `severity_weighted_sum` | float | 0–2+ | mild=0.33, moderate=0.66, severe=1.0, summed |

**Use:** Severe symptoms (especially chest pain, dyspnea) strongly increase risk.

---

## 4. Duration / Temporal Features

| Feature | Type | Description |
|---------|------|-------------|
| `max_duration_days` | int/float | Longest duration in days (parsed from "3 days", "2 weeks") |
| `acute_flag` | binary | 1 if any symptom < 24 hours |
| `chronic_flag` | binary | 1 if any symptom > 2 weeks |
| `duration_unknown` | binary | 1 if no duration mentioned |

**Use:** Acute onset + severe symptoms → higher urgency. Chronic alone may be lower urgency.

---

## 5. Red-Flag / High-Risk Symptoms

Symptoms that typically require urgent attention:

| Symptom | SNOMED (example) | Risk weight |
|---------|------------------|-------------|
| Chest pain | 29857009 | High |
| Shortness of breath | 267036007 | High |
| Coughing blood | (hemoptysis) | High |
| Blood in stool | (rectal bleeding) | High |
| Severe abdominal pain | - | High |
| Neurological (blurred vision, severe headache) | - | Medium-High |

| Feature | Type | Description |
|---------|------|-------------|
| `has_red_flag` | binary | 1 if any red-flag symptom present |
| `red_flag_count` | int | Number of red-flag symptoms |
| `red_flag_severe` | binary | 1 if red-flag + severe |

**Use:** Strong boost to risk score when present.

---

## 6. Symptom Combinations (Syndromes)

Common clinical patterns:

| Combination | Example conditions | Risk tendency |
|-------------|--------------------|---------------|
| Fever + cough + fatigue | Flu, COVID, pneumonia | Medium |
| Chest pain + shortness of breath | Cardiac, pulmonary | High |
| Fever + rash | Viral, allergic | Medium |
| Abdominal pain + vomiting + diarrhea | Gastroenteritis | Medium |
| Cough + blood | TB, lung cancer | High |
| Fever + headache + neck stiffness | Meningitis | High |

| Feature | Type | Description |
|---------|------|-------------|
| `syndrome_cardiac_like` | binary | chest_pain + shortness_of_breath |
| `syndrome_respiratory` | binary | fever + cough + fatigue |
| `syndrome_gi` | binary | abdominal_pain + (vomiting or diarrhea) |
| `syndrome_alarm` | binary | coughing_blood or blood_in_stool |

**Use:** Syndrome presence can override simple symptom counts.

---

## 7. Demographics

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Patient age (impute 40 if missing) |
| `age_group` | categorical | 0=<18, 1=18–40, 2=40–60, 3=60+ |
| `gender` | categorical | male, female, other, unknown |
| `pregnancy` | binary | If relevant (from history) |

**Use:** Age and gender affect risk for certain conditions (e.g. cardiac risk higher in older males).

---

## 8. Medical History

| Feature | Type | Description |
|---------|------|-------------|
| `has_diabetes` | binary | From history |
| `has_hypertension` | binary | From history |
| `has_heart_disease` | binary | From history |
| `has_lung_disease` | binary | From history |
| `has_immunocompromise` | binary | From history |
| `comorbidity_count` | int | Number of chronic conditions |

**Use:** Comorbidities increase risk and change interpretation of symptoms.

---

## 9. Feature Vector for ML (Flat)

For XGBoost/LightGBM, flatten to:

```
# Symptom presence (one per symptom in ontology)
has_fever, has_chest_pain, has_cough, has_shortness_of_breath, ...

# Counts
symptom_count, severe_count, red_flag_count

# Severity
max_severity_encoded, has_severe, severity_weighted_sum

# Duration
max_duration_days, acute_flag, chronic_flag

# Red flags
has_red_flag, red_flag_severe

# Syndromes
syndrome_cardiac_like, syndrome_respiratory, syndrome_gi, syndrome_alarm

# Demographics
age, age_group, gender_encoded

# History
has_diabetes, has_hypertension, comorbidity_count
```

---

## 10. Target Variables (for Training)

| Target | Type | Description |
|--------|------|-------------|
| `risk_level` | categorical | LOW, MODERATE, HIGH |
| `risk_score` | float | 0–1 continuous |
| `urgency` | categorical | routine, urgent, emergency |
| `possible_conditions` | list[str] | Differential diagnosis (from symptom-disease mapping) |

---

## 11. Data Sources for Training

| Source | Use |
|--------|-----|
| Symptom–disease datasets | Map symptoms → conditions |
| Triage datasets | Map features → urgency/risk |
| Clinical notes (annotated) | Learn risk from real cases |
| Rule-based labels | Bootstrap when no labeled data |

---

## 12. Summary — Feature Checklist

| # | Feature | Category | Priority |
|---|---------|----------|----------|
| 1 | Symptom presence (one-hot) | Symptom | High |
| 2 | symptom_count | Symptom | High |
| 3 | max_severity, has_severe, severe_count | Severity | High |
| 4 | max_duration_days, acute_flag | Duration | Medium |
| 5 | has_red_flag, red_flag_count | Red-flag | High |
| 6 | syndrome_* (cardiac, respiratory, gi, alarm) | Syndrome | High |
| 7 | age, age_group, gender | Demographics | Medium |
| 8 | has_diabetes, has_hypertension, comorbidity_count | History | Medium |
