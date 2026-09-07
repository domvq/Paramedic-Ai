import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("model.joblib")

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
    return joblib.load(MODEL_PATH)

def predict_deterioration(
    age,
    heart_rate,
    systolic_bp,
    diastolic_bp,
    respiratory_rate,
    spo2,
    temperature
):
    model = load_model()

    patient = pd.DataFrame([{
        "age": age,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "spo2": spo2,
        "temperature": temperature
    }])

    probability = model.predict_proba(patient[FEATURES])[0][1]

    return probability

if __name__ == "__main__":
    probability = predict_deterioration(
        age=65,
        heart_rate=110,
        systolic_bp=90,
        diastolic_bp=55,
        respiratory_rate=26,
        spo2=92,
        temperature=38.5
    )

    print()
    print("Paramedic AI demo prediction")
    print("----------------------------")
    print(f"Estimated probability: {probability:.1%}")

    if probability >= 0.50:
        print("Demo result: HIGHER RISK")
    else:
        print("Demo result: LOWER RISK")

    print()
    print("DEMO ONLY - NOT FOR CLINICAL DECISION MAKING")
