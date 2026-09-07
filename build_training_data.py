import pandas as pd
from pathlib import Path

BASE = Path("data/mimic-iv-clinical-database-demo-2.2")
ICU = BASE / "icu"
HOSP = BASE / "hosp"

print("Loading deterioration data...")
vitals = pd.read_csv("data/deterioration_data.csv")

print(f"Vital records: {len(vitals)}")

print("Loading admissions...")
admissions = pd.read_csv(HOSP / "admissions.csv.gz")

# Keep only the fields needed for the outcome
outcomes = admissions[
    ["subject_id", "hadm_id", "hospital_expire_flag"]
].copy()

# Merge outcome onto the vital-sign records
data = vitals.merge(
    outcomes,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Missing outcome means the admission was not found.
data["hospital_expire_flag"] = data["hospital_expire_flag"].fillna(0).astype(int)

# Our prototype target:
# 1 = patient died during the hospitalization
# 0 = patient survived hospitalization
data["deterioration"] = data["hospital_expire_flag"]

# Remove records with missing model features
features = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
]

data = data.dropna(subset=features)

print("\nTraining dataset:")
print(data[features + ["deterioration"]].to_string(index=False))

print("\nClass counts:")
print(data["deterioration"].value_counts())

# Save final dataset
output = Path("data/training_data.csv")
data.to_csv(output, index=False)

print(f"\nSaved: {output}")
print(f"Rows: {len(data)}")
print("Done.")