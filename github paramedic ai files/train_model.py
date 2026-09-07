import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = Path("data/deterioration_data.csv")
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

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", list(df.columns))

X = df[FEATURES]
y = df["hospital_expire_flag"]

print("\nOutcome counts:")
print(y.value_counts())

if y.nunique() < 2:
    raise RuntimeError("The dataset does not contain both outcome classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )),
])

print("\nTraining model...")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Test accuracy:", round(accuracy, 3))

print("\nSaving model...")
joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH.resolve())
print("Training complete!")
