# Symptom Examples — Best Coverage

Symptom phrases that work well with NER extraction, triage mapper, and risk model. These cover most common clinical presentations.

---

## 1. Respiratory / Flu-like (most common)

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "I have fever and headache" | Fever and mild headache | Very common |
| "Fever, cough, and fatigue" | Cough and cold | Triggers syndrome_respiratory |
| "Cough and cold for a week" | Cough and cold | Duration parsed |
| "Sore throat and runny nose" | Cold and throat irritation | |
| "Persistent cough for 2 weeks" | Persistent cough | |
| "Mild fever and body ache" | Mild fever | |
| "Low grade fever" | Low grade fever | |
| "High fever for 5 days" | High fever for 5 days | Needs duration 4–7 days |

**Covers:** Flu, common cold, mild COVID-like illness, upper respiratory infection.

---

## 2. Gastrointestinal

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "Stomach pain and vomiting" | Stomach pain and vomiting | Triggers syndrome_gi |
| "Abdominal pain and diarrhea" | Stomach pain and vomiting | |
| "Vomiting and dehydration" | Vomiting and dehydration | |
| "Severe abdominal pain" | Severe abdominal pain | Needs "severe" |
| "Nausea and vomiting" | Stomach pain and vomiting | Nausea → GI pattern |

**Covers:** Gastroenteritis, food poisoning, mild GI upset.

---

## 3. Cardiac / Urgent (red-flag)

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "Chest pain and shortness of breath" | Severe chest pain or Chest tightness and sweating | Triggers syndrome_cardiac_like |
| "Chest tightness and sweating" | Chest tightness and sweating | |
| "Shortness of breath" | Breathing shortness | |
| "Severe shortness of breath" | Severe breathing difficulty | Needs "severe" |
| "Severe chest pain" | Severe chest pain | |
| "Chest pain, sweating, feeling breathless" | Chest tightness and sweating | |

**Covers:** Possible cardiac event, pulmonary embolism, severe anxiety.

---

## 4. Headache / Neurological

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "Headache for a week" | Headache since one week | |
| "Severe headache" | Severe migraine | |
| "Migraine" | Severe migraine | |
| "Mild headache" | Fever and mild headache (if fever) else Mild fever | |

**Covers:** Tension headache, migraine, post-viral headache.

---

## 5. Mild / Low-urgency

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "Mild back pain" | Mild back pain | |
| "Skin rash and itching" | Mild skin rash | |
| "High blood pressure symptoms" | High blood pressure symptoms | |
| "Allergic reaction, hives" | Severe allergic reaction | |

**Covers:** Musculoskeletal, dermatological, chronic conditions.

---

## 6. Alarm / High-risk

| Example phrase | Triage category | Notes |
|----------------|-----------------|-------|
| "Coughing blood" | (alarm syndrome) | Triggers syndrome_alarm |
| "Blood in stool" | (alarm syndrome) | |
| "Hemoptysis" | (alarm syndrome) | In lexicon |

**Covers:** TB, lung cancer, GI bleed — typically Emergency.

---

## 7. Phrases That Map to Default

If no rule matches → **Cough and cold** (safest default).

| Example | Why default |
|---------|-------------|
| "General discomfort" | No specific keywords |
| "Feeling unwell" | Too vague |
| "Dizziness and fatigue" | No direct mapping (still extracted) |

---

## 8. Best Coverage — Top 10 Combinations

Use these for testing or to cover most real-world cases:

1. **"I have fever and headache"** — flu-like
2. **"Fever, cough, and fatigue"** — respiratory syndrome
3. **"Chest pain and shortness of breath"** — cardiac-like
4. **"Stomach pain and vomiting"** — GI
5. **"Cough and cold for a few days"** — common cold
6. **"Severe headache"** — migraine
7. **"Sore throat and runny nose"** — throat/cold
8. **"Mild back pain"** — musculoskeletal
9. **"High fever for 5 days"** — prolonged fever (with duration)
10. **"Chest tightness and sweating"** — possible cardiac

---

## 9. Lexicon Variations (NER will normalize)

The symptom lexicon maps variations to canonical names:

| You say | Canonical |
|---------|-----------|
| feverish, pyrexia, high temperature | fever |
| breathlessness, dyspnea, can't breathe | shortness of breath |
| stomach pain, belly pain, stomachache | abdominal pain |
| throwing up, puking, emesis | vomiting |
| head pain, migraine, cephalgia | headache |
| chest discomfort, chest tightness, chest pressure | chest pain |

Use any of these — NER will normalize before feature building.
