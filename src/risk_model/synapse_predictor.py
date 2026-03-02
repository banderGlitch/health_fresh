"""
SYNAPSE Triage Predictor — ML-based risk model.

Uses the SYNAPSE-trained RandomForest to predict OTC vs Doctor Consultation.
Maps pipeline features (symptoms, demographics, duration, severity) to SYNAPSE
input format and converts prediction to risk_score and Severity.

Input format (from train_model.py):
  - Symptoms: comma-separated text (TfidfVectorizer)
  - Gender, Age, Duration, Severity: encoded integers, scaled with StandardScaler

Encoding (LabelEncoder, alphabetical):
  - Gender: Female=0, Male=1
  - Age: 16-45=0, 16-60=1, 6-15=2, above 45=3, above 60=4, below 5=5
  - Duration: Greater than 3 days=0, Less than 3 days=1
  - Severity: Mild=0, Moderate=1, Severe=2
"""

from pathlib import Path
from typing import Any

# Project root: src/risk_model/ -> project root is parent of src
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "models" / "synapse_triage"

# --- Encoding helpers: pipeline format -> SYNAPSE model format ---


def _age_to_synapse_encoded(age: int) -> int:
    """
    Map numeric age to SYNAPSE Age encoding.
    SYNAPSE: 0=16-45, 1=16-60, 2=6-15, 3=above 45, 4=above 60, 5=below 5
    """
    if age < 5:
        return 5  # below 5 years
    if age <= 15:
        return 2  # 6-15 years
    if age <= 45:
        return 0  # 16-45 years
    if age <= 60:
        return 1  # 16-60 years (covers 46-60)
    return 4  # above 60 years


def _gender_to_synapse_encoded(gender: str) -> int:
    """Map gender string to SYNAPSE encoding: Female=0, Male=1."""
    g = str(gender).lower().strip()
    return 1 if g in ("male", "m") else 0  # Male=1, else Female=0


def _duration_to_synapse_encoded(max_days: float | None, acute_flag: int) -> int:
    """
    Map duration to SYNAPSE encoding.
    0 = Greater than 3 days, 1 = Less than 3 days
    """
    if acute_flag:
        return 1  # Less than 3 days (acute = short)
    if max_days is None:
        return 0  # Unknown -> treat as longer
    return 1 if max_days <= 3 else 0


def _severity_to_synapse_encoded(severities: list[str | None]) -> int:
    """Map severity to SYNAPSE encoding: Mild=0, Moderate=1, Severe=2. Uses worst."""
    mapping = {"mild": 0, "moderate": 1, "severe": 2}
    max_enc = 0
    for s in (severities or []):
        if s:
            max_enc = max(max_enc, mapping.get(str(s).lower(), 1))
    return max_enc


def _symptoms_to_synapse_text(features: dict[str, Any]) -> str:
    """
    Get comma-separated symptoms string from features.
    FeatureBuilder provides symptom_text. Fallback to avoid empty input.
    """
    return features.get("symptom_text") or "general discomfort"


class SynapseTriagePredictor:
    """
    Risk predictor backed by SYNAPSE-trained RandomForest.

    Loads model, vectorizer, scaler, label_encoder from models/synapse_triage/.
    Adapts pipeline features to SYNAPSE input and maps output to risk_score/Severity.
    """

    def __init__(self, model_dir: Path | str | None = None):
        """
        Load SYNAPSE model artifacts.

        Args:
            model_dir: Override path to model folder. Default: models/synapse_triage
        """
        self._model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._model = None
        self._vectorizer = None
        self._scaler = None
        self._label_encoder = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load model, vectorizer, scaler, label_encoder from disk."""
        try:
            import joblib
            from scipy.sparse import hstack, csr_matrix

            self._model = joblib.load(self._model_dir / "telehealth_model.pkl")
            self._vectorizer = joblib.load(self._model_dir / "vectorizer.pkl")
            self._scaler = joblib.load(self._model_dir / "scaler.pkl")
            self._label_encoder = joblib.load(self._model_dir / "label_encoder.pkl")
            self._hstack = hstack
            self._csr_matrix = csr_matrix
            self._loaded = True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False

    @property
    def is_available(self) -> bool:
        """True if model loaded successfully."""
        return self._loaded

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Run SYNAPSE triage prediction.

        Maps features (from FeatureBuilder) to SYNAPSE input, runs model,
        converts OTC/Doctor to RiskScore and Severity.

        Args:
            features: Dict from FeatureBuilder.build() with keys:
                - symptom names (via extraction_dict), severities, durations,
                  demographics (age, gender), acute_flag, max_duration_days

        Returns:
            Dict with RiskScore, Severity, Confidence, possible_conditions,
            triage_recommendation (OTC Drug | Doctor Consultation)
        """
        if not self._loaded:
            return self._fallback_output(
                "SYNAPSE model not loaded",
                features,
            )

        # Step 1: Extract pipeline features -> SYNAPSE input format
        symptoms_text = _symptoms_to_synapse_text(features)

        age = features.get("age", 40)
        gender = features.get("gender", "unknown")
        severities = features.get("severities") or []
        max_duration_days = features.get("max_duration_days")
        acute_flag = features.get("acute_flag", 0)

        gender_enc = _gender_to_synapse_encoded(gender)
        age_enc = _age_to_synapse_encoded(age)
        duration_enc = _duration_to_synapse_encoded(max_duration_days, acute_flag)
        severity_enc = _severity_to_synapse_encoded(severities)

        # Step 2: Vectorize symptoms (TF-IDF) + scale numeric features; combine into single matrix
        import numpy as np

        text_feat = self._vectorizer.transform([symptoms_text])
        numeric = np.array([[gender_enc, age_enc, duration_enc, severity_enc]])
        numeric_scaled = self._scaler.transform(numeric)
        X = self._hstack([text_feat, self._csr_matrix(numeric_scaled)])

        # Step 3: Run RandomForest -> 0=Doctor, 1=OTC (label_encoder inverse)
        pred_enc = self._model.predict(X)[0]
        triage = self._label_encoder.inverse_transform([pred_enc])[0]

        # Step 4: Map SYNAPSE output (OTC/Doctor) -> risk_score and Severity for pipeline
        if triage == "OTC Drug":
            risk_score = 0.3
            severity = "LOW"
        else:
            risk_score = 0.8
            severity = "HIGH"

        # Safety: override to HIGH if red-flag + severe (model may miss edge cases)
        if features.get("has_red_flag") and features.get("red_flag_severe"):
            risk_score = 0.95
            severity = "HIGH"

        return {
            "RiskScore": round(risk_score, 2),
            "Severity": severity,
            "Confidence": 0.85,
            "possible_conditions": self._possible_conditions(features),
            "triage_recommendation": triage,
        }

    def _fallback_output(
        self,
        reason: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Heuristic fallback when model unavailable."""
        severity = "MODERATE"
        risk_score = 0.5
        if features.get("has_severe"):
            severity, risk_score = "HIGH", 0.8
        elif features.get("severities") and features.get("severities")[0] == "mild":
            severity, risk_score = "LOW", 0.3
        if features.get("has_red_flag"):
            risk_score = min(1.0, risk_score + 0.15)
        return {
            "RiskScore": round(risk_score, 2),
            "Severity": severity,
            "Confidence": 0.5,
            "possible_conditions": [],
            "triage_recommendation": "Unknown (fallback)",
        }

    def _possible_conditions(self, features: dict[str, Any]) -> list[str]:
        """Map syndrome features to possible conditions."""
        conditions = []
        if features.get("syndrome_respiratory"):
            conditions.append("Respiratory infection (consider flu, COVID, pneumonia)")
        if features.get("syndrome_cardiac_like"):
            conditions.append("Cardiac or pulmonary consideration")
        if features.get("syndrome_gi"):
            conditions.append("Gastrointestinal condition")
        return conditions
