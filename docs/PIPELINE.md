# AI-Analyzer Pipeline Architecture

> **By CepiaLabs** | Medical Diagnosis App Pipeline

This document describes the end-to-end pipeline for the AI-Analyzer module, aligned with the [Krish Naik Medical Diagnosis App](https://www.krishnaik.in/project/medical-diagnosis-app) reference.

---

## Objectives

The module must:

1. **Extract symptoms precisely** — Identify medical facts from natural language
2. **Normalize to medical ontology** — Map to standard medical vocabularies (e.g., SNOMED CT)
3. **Capture context** — Duration, severity, triggers, associated factors
4. **Infer possible conditions** — Probabilistic inference for likely diagnoses
5. **Estimate risk & uncertainty** — Risk scores and confidence levels

---

## End-to-End Flow

```
Conversation → NER Extraction → Ontology Mapping → Feature Builder → Risk Model → LLM Clarification → Final Structured Symptom Profile
```

---

## Pipeline Stages (Detailed)

### 1. Structured NLP Extraction (Deterministic + ML)

**What it does:** Pulls structured medical facts from conversation.

| Extract | Examples |
|---------|----------|
| **Symptoms** | fever, chest pain |
| **Duration** | 3 days |
| **Severity** | mild / moderate / severe |
| **Associated factors** | nausea, sweating |
| **Negatives** | "no shortness of breath" |

**Techniques:**
- **Medical NER model** — Named Entity Recognition for symptoms, diseases, body parts
- **Pattern rules** — For units & time (e.g., "3 days", "twice daily")
- **Negation detection** — Identify absent symptoms

**Output Example:**
```json
{
  "symptoms": [
    { "name": "fever", "duration": "3 days", "severity": "moderate" },
    { "name": "headache", "severity": "mild" }
  ],
  "negated": ["vomiting"]
}
```

**Phase 1 Status:** Implemented — symptom lexicon, pattern rules, negation detection.

---

### 2. Ontology Mapping Layer

**Why:** Patients use varied terms ("burning chest", "tightness", "pressure") — all should map to standard concepts.

**Methods (medical vocabularies):**
- **Symptom normalization** — Canonical form for each symptom
- **Synonym expansion** — Link equivalent terms
- **Hierarchical relationships** — Broader/narrower medical concepts

**Example output:** `SNOMED_CODE: 386661006 (Fever)`

---

### 3. Feature Builder

**Inputs:**
- Symptom vectors (from ontology mapping)
- Demographics (age, gender, location)
- History (medical history, temporal patterns)

**Output:** Feature matrix for the risk model.

---

### 4. Probabilistic Inference Model (Risk Model)

**Predicts:**
- Likely condition categories
- Severity risk
- Confidence

**Model types:**
- **Gradient boosting / deep tabular** — XGBoost, LightGBM
- **Bayesian network** — Great for clinical reasoning and uncertainty

**Output Example:**
```
RiskScore: 0.78
Severity: MODERATE
```

---

### 5. LLM Clinical Reasoning Layer

**Role:** Handle nuance and ambiguity:
- Vague descriptions
- Multi-symptom reasoning
- Clarifying questions

**Guardrails:**
- **LLM cannot finalize severity** → Must pass through risk engine
- LLM suggests; risk model decides

---

### 6. Final Structured Symptom Profile

Combined output: standardized symptoms, ontology codes, risk scores, and LLM-generated clarifications.

---

## Datasets You Can Use

| Category | Datasets | Use For |
|----------|----------|---------|
| **Symptom Extraction / NER** | Clinical text corpora, medical conversation datasets, annotated symptom mentions | Training NER models, negation detection |
| **Symptom → Condition Mapping** | Symptom-disease association datasets, triage datasets | Mapping symptoms to conditions |

---

## Project Structure (Recommended)

```
cepialabs_health_care/
├── src/
│   ├── extraction/      # Stage 1: NER + pattern rules
│   ├── ontology/        # Stage 2: SNOMED mapping
│   ├── features/        # Stage 3: Feature builder
│   ├── risk_model/      # Stage 4: Probabilistic inference
│   ├── llm_reasoning/   # Stage 5: LLM clarification
│   └── pipeline.py      # Orchestrates all stages
├── api/                 # FastAPI endpoints
├── models/              # Pydantic schemas
└── config/              # Configuration
```
