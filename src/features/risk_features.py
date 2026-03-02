"""
Risk model feature definitions.

Red-flag symptoms, syndromes, severity encoding.
Used by FeatureBuilder to construct the feature vector.
"""

# Symptoms that typically indicate higher urgency
RED_FLAG_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "coughing blood",
    "blood in stool",
    "hemoptysis",
    "rectal bleeding",
}

# Syndrome: symptom sets indicating clinical patterns
# Flagged when at least one symptom from set is present
SYNDROME_DEFINITIONS = {
    "cardiac_like": {"chest pain", "shortness of breath"},
    "respiratory": {"fever", "cough", "fatigue"},
    "gi": {"abdominal pain", "vomiting", "diarrhea"},
    "alarm": {"coughing blood", "blood in stool"},
}

# Severity encoding for ML
SEVERITY_ENCODING = {"mild": 0.33, "moderate": 0.66, "severe": 1.0}
