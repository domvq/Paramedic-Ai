import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    "models",
    "deterioration_model.pkl"
)


FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature"
]


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Deterioration model has not been trained yet."
        )

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

    probability = model.predict_proba(
        patient[FEATURES]
    )[0][1]

    return probability