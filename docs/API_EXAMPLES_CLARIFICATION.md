# API Examples: Clarification Flow (with LLM Merge)

Run the API: `uvicorn api.main:app --reload` (from project root)

Ensure `.env` has `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` for LLM merge.

---

## Step 1: POST /analyze (initial complaint)

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": "I have fever and headache",
    "demographics": {"age": 35, "gender": "male"}
  }'
```

**Expected logs:**
```
[ANALYZE] conversation=20 chars, demographics={'age': 35, 'gender': 'male'}
[NER] Extracted ...
[ONTOLOGY] Mapped ...
[FEATURES] ...
[PREDICTION] RiskScore=... Severity=... Triage=...
[ANALYZE] session_id=abc123..., triage=OTC Drug, questions=3
```

**Response:** Save `session_id` from the JSON response.

---

## Step 2: POST /analyze/continue (answer clarifying questions)

**Request:** Use the `session_id` from Step 1.
```bash
curl -X POST http://localhost:8000/analyze/continue \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "answers": "3 days, mild, no other symptoms"
  }'
```

**Expected logs (LLM merge path):**
```
[CONTINUE] session=abc12345, answers=31 chars, stored_questions=3
[CONTINUE] Using LLM merge
[CLARIFICATION] Merge requested: original=20 chars, 3 questions, answers=31 chars
[CLARIFICATION] LLM merged output (87 chars): Patient has fever and mild headache for 3 days. No other symptoms.
[CONTINUE] Re-running pipeline on merged text (87 chars)
[NER] Extracted ...
[PREDICTION] ...
[CONTINUE] Done. triage=OTC Drug, severity=LOW
```

**If LLM unavailable (fallback):**
```
[CONTINUE] Fallback: simple append (LLM=False, questions=3)
[CONTINUE] Re-running pipeline on merged text (...)
```

---

## Full Example (PowerShell)

```powershell
# 1. Analyze
$r1 = Invoke-RestMethod -Uri "http://localhost:8000/analyze" -Method POST `
  -ContentType "application/json" `
  -Body '{"conversation": "I have fever and headache", "demographics": {"age": 35, "gender": "male"}}'

Write-Host "Session:" $r1.session_id
Write-Host "Triage:" $r1.triage_recommendation
Write-Host "Questions:" ($r1.llm_clarification.clarifying_questions -join " | ")

# 2. Continue with answers
$r2 = Invoke-RestMethod -Uri "http://localhost:8000/analyze/continue" -Method POST `
  -ContentType "application/json" `
  -Body (@{ session_id = $r1.session_id; answers = "3 days, mild, no" } | ConvertTo-Json)

Write-Host "After clarification - Triage:" $r2.triage_recommendation "Severity:" $r2.severity
```

---

## More Test Cases (Easy)

| Step 1 (conversation) | Step 2 (answers) | Expected triage |
|-----------------------|------------------|-----------------|
| "I have fever and headache" | "3 days, mild, no" | OTC Drug |
| "Chest pain and sweating" | "severe, started yesterday" | Emergency |
| "Cough and runny nose" | "about a week, mild" | OTC Drug |
| "High fever" | "5 days, child 8 years" | Doctor Consultation |

---

## Tough / Complex Test Cases

### 1. Multiple symptoms, different durations
**Step 1:**
```json
{"conversation": "I have fever, headache, and a cough. Also some stomach discomfort.", "demographics": {"age": 42, "gender": "female"}}
```
**Step 2:** `"The fever and headache started 3 days ago. Cough has been there for about 2 weeks. Stomach pain is mild, started yesterday."`  
**Why tough:** Multiple durations per symptom; LLM must link each duration to the right symptom.

---

### 2. Ambiguous short answers
**Step 1:**
```json
{"conversation": "Chest tightness and shortness of breath", "demographics": {"age": 58, "gender": "male"}}
```
**Step 2:** `"yes, 2 days, severe"`  
**Why tough:** "yes" is vague; LLM must infer it relates to severity or confirmation. "2 days" and "severe" must map to duration and severity.

---

### 3. Medical jargon + lay mix
**Step 1:**
```json
{"conversation": "Experiencing dyspnea and diaphoresis", "demographics": {"age": 55, "gender": "female"}}
```
**Step 2:** `"Started this morning. Very bad. No chest pain but feel weak."`  
**Why tough:** Original uses medical terms; answers are lay. LLM should merge to NER-friendly phrasing (shortness of breath, sweating).

---

### 4. Contradictory / partial answers
**Step 1:**
```json
{"conversation": "High fever and persistent cough", "demographics": {"age": 8, "gender": "male"}}
```
**Step 2:** `"Fever for 5 days. Cough I'm not sure, maybe 3 days? Moderate."`  
**Why tough:** Different durations for each symptom; "moderate" could apply to one or both.

---

### 5. Long-form narrative answer
**Step 1:**
```json
{"conversation": "Abdominal pain and vomiting", "demographics": {"age": 35, "gender": "female"}}
```
**Step 2:** `"The stomach pain has been severe since yesterday evening. I've vomited 4 times. No fever. Haven't been able to keep food down."`  
**Why tough:** Answer is a narrative; LLM must distill to structured symptom + duration + severity.

---

### 6. Negative / exclusionary answers
**Step 1:**
```json
{"conversation": "Headache and dizziness", "demographics": {"age": 50, "gender": "male"}}
```
**Step 2:** `"Headache for a week, moderate. No chest pain. No shortness of breath. Dizzy on and off."`  
**Why tough:** Negations ("no X") should be preserved; "on and off" is vague for duration.

---

### 7. Pediatric, mixed severity
**Step 1:**
```json
{"conversation": "My 6 year old has fever, cough, and runny nose", "demographics": {"age": 6, "gender": "female"}}
```
**Step 2:** `"High fever for 4 days. Cough is mild, maybe 5 days. Runny nose for a week. She's drinking okay, no rash."`  
**Why tough:** Multiple symptoms with different durations and severities; age is critical for triage.

---

### 8. Free-form messy answer
**Step 1:**
```json
{"conversation": "Chest pain, sweating, feeling breathless", "demographics": {"age": 62, "gender": "male"}}
```
**Step 2:** `"umm like 1-2 days? pretty severe. yes sweating a lot. breathlessness worse when i move"`  
**Why tough:** Informal, fragmented; "umm", "like", "yes" add noise; LLM must extract 1-2 days, severe.

---

## Curl Examples (Tough Cases)

```bash
# Case 2: Ambiguous short answers
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"conversation": "Chest tightness and shortness of breath", "demographics": {"age": 58, "gender": "male"}}'
# Use session_id from response, then:
curl -X POST http://localhost:8000/analyze/continue -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "answers": "yes, 2 days, severe"}'

# Case 5: Long-form narrative
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"conversation": "Abdominal pain and vomiting", "demographics": {"age": 35, "gender": "female"}}'
curl -X POST http://localhost:8000/analyze/continue -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "answers": "The stomach pain has been severe since yesterday evening. I have vomited 4 times. No fever. Cannot keep food down."}'

# Case 8: Messy informal
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"conversation": "Chest pain, sweating, feeling breathless", "demographics": {"age": 62, "gender": "male"}}'
curl -X POST http://localhost:8000/analyze/continue -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "answers": "umm like 1-2 days? pretty severe. yes sweating a lot. breathlessness worse when i move"}'
```
