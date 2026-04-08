/**
 * Symptoms the backend NER can extract — every string must appear in
 * `src/extraction/symptom_lexicon.py` (canonical name or a listed variation).
 *
 * The chat UI only allows picking from this set (no free-typed symptom lists).
 */
export const SYMPTOMS_BY_CATEGORY = {
  General: [
    "fever",
    "headache",
    "chills",
    "feeling sick",
    "feeling tired",
    "body ache",
    "sweating",
    "loss of appetite",
    "insomnia",
    "weight loss",
    "joint pain",
    "back pain",
    "swelling",
  ],
  Respiratory: [
    "cough",
    "sore throat",
    "shortness of breath",
    "runny nose",
    "coughing blood",
  ],
  "Chest & heart": ["chest pain", "palpitations"],
  Abdominal: ["abdominal pain", "vomiting", "diarrhea"],
  "Neurological & senses": [
    "dizziness",
    "blurred vision",
    "ear pain",
    "eye pain",
  ],
  Skin: ["rash"],
  Other: ["blood in stool"],
};

/** Flat list for inline pickers and validation (order: category order). */
export const ALLOWED_SYMPTOMS_FLAT = Object.values(SYMPTOMS_BY_CATEGORY).flat();

/** Set for quick membership checks */
export const ALLOWED_SYMPTOM_SET = new Set(
  ALLOWED_SYMPTOMS_FLAT.map((s) => s.toLowerCase())
);

export const DURATIONS = [
  { value: "2", label: "Less than 3 days", acute: true },
  { value: "5", label: "3–7 days", acute: false },
  { value: "14", label: "More than 1 week", acute: false },
];

export const SEVERITIES = ["Mild", "Moderate", "Severe"];
