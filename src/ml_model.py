from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_risk(
    model,
    age,
    heart_rate,
    systolic_bp,
    diastolic_bp,
    respiratory_rate,
    spo2,
    temperature,
):
    patient = pd.DataFrame(
        [
            {
                "age": age,
                "heart_rate": heart_rate,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "respiratory_rate": respiratory_rate,
                "spo2": spo2,
                "temperature": temperature,
            }
        ]
    )

    patient = patient[FEATURES]

    probability = model.predict_proba(patient)[0][1]

    return float(probability)


def classify_risk(probability):
    if probability >= 0.70:
        return "HIGHER RISK"

    if probability >= 0.40:
        return "INTERMEDIATE RISK"

    return "LOWER RISK"
