/**
 * Curated symptoms — only those the RAG/SYNAPSE model handles well.
 * Derived from SYNAPSE dataset and test_dataset_cases.py.
 */
export const SYMPTOMS_BY_CATEGORY = {
  "General": [
    "fever",
    "headache",
    "itching",
    "flu like illness",
    "feeling sick",
    "feeling tired",
    "feeling unwell",
    "chills",
  ],
  "Respiratory": [
    "cough",
    "sore throat",
    "shortness of breath",
    "runny nose",
    "sneezing",
    "coughing blood",
    "whoop sound after coughing",
    "pause in breathing",
  ],
  "Abdominal": [
    "abdominal pain",
    "vomiting",
    "constipation",
    "diarrhea",
    "bloating",
  ],
  "Chest & Heart": [
    "chest pain",
    "fast heart rate",
    "feeling lightheaded",
    "sensation of spinning",
  ],
  "Skin": [
    "skin pain",
    "skin color change",
    "itchy skin rash",
    "blotchy",
    "skin redness",
    "swelling skin",
    "darkening of skin",
    "red firm lumps",
    "scaly patches with irregular borders",
  ],
  "Eyes & Vision": [
    "blurred vision",
    "sensitivity to light",
    "seeing rainbows around lights",
  ],
  "Urinary": ["blood urine", "painful urination"],
  "Other": [
    "neck pain",
    "unwanted weight loss",
    "lump on neck",
    "hoarse voice",
    "pain with swallowing",
    "unexplained ear discomfort",
    "pins needles feeling in fingers",
    "pins needles feeling in toes",
    "garlic odor in breath",
    "abnormal hair growth",
    "new purplish stretch marks",
    "swelling leg",
    "false beliefs",
    "seeing hearing things not there",
  ],
};

export const DURATIONS = [
  { value: "2", label: "Less than 3 days", acute: true },
  { value: "5", label: "3–7 days", acute: false },
  { value: "14", label: "More than 1 week", acute: false },
];

export const SEVERITIES = ["Mild", "Moderate", "Severe"];
