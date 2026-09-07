import os
import pandas as pd


BASE = "data/mimic-iv-clinical-database-demo-2.2"

ICU = os.path.join(
    BASE,
    "icu"
)

HOSP = os.path.join(
    BASE,
    "hosp"
)

OUTPUT = os.path.join(
    "data",
    "deterioration_data.csv"
)


# MIMIC-IV item IDs
ITEMS = {
    "heart_rate": 220045,
    "respiratory_rate": 220210,
    "spo2": 220277,
    "systolic_bp": 220179,
    "diastolic_bp": 220180,
    "temperature": 223762
}


def main():

    print("Loading patients...")

    patients = pd.read_csv(
        os.path.join(
            HOSP,
            "patients.csv.gz"
        )
    )

    print(
        f"Patients loaded: {len(patients)}"
    )


    print("Loading ICU stays...")

    icustays = pd.read_csv(
        os.path.join(
            ICU,
            "icustays.csv.gz"
        )
    )

    print(
        f"ICU stays loaded: {len(icustays)}"
    )


    print("Loading chart events...")

    chartevents = pd.read_csv(
        os.path.join(
            ICU,
            "chartevents.csv.gz"
        ),
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "itemid",
            "valuenum"
        ]
    )

    print(
        f"Chart events loaded: {len(chartevents)}"
    )


    # Keep only the vital signs we need
    vital_ids = list(
        ITEMS.values()
    )

    vitals = chartevents[
        chartevents["itemid"].isin(
            vital_ids
        )
    ].copy()


    print(
        f"Vital-sign records: {len(vitals)}"
    )


    # Convert chart time
    vitals["charttime"] = pd.to_datetime(
        vitals["charttime"]
    )


    # Map MIMIC item IDs to our feature names
    id_to_name = {
        value: key
        for key, value in ITEMS.items()
    }

    vitals["feature"] = vitals[
        "itemid"
    ].map(id_to_name)


    # Remove missing measurements
    vitals = vitals.dropna(
        subset=["valuenum"]
    )


    # Sort by patient and time
    vitals = vitals.sort_values(
        [
            "stay_id",
            "charttime"
        ]
    )


    print("Selecting initial vital signs...")


    # First available measurement for
    # each vital sign during each ICU stay.
    initial = (
        vitals
        .groupby(
            [
                "stay_id",
                "feature"
            ],
            as_index=False
        )
        .first()
    )


    # Convert from long format to wide format
    dataset = initial.pivot(
        index="stay_id",
        columns="feature",
        values="valuenum"
    ).reset_index()


    # Remove column index name
    dataset.columns.name = None


    # Add ICU admission information
    dataset = dataset.merge(
        icustays[
            [
                "stay_id",
                "subject_id",
                "hadm_id",
                "intime",
                "outtime"
            ]
        ],
        on="stay_id",
        how="left"
    )


    # Add patient age
    dataset = dataset.merge(
        patients[
            [
                "subject_id",
                "anchor_age"
            ]
        ],
        on="subject_id",
        how="left"
    )


    dataset = dataset.rename(
        columns={
            "anchor_age": "age"
        }
    )


    # Keep only the model variables
    columns = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "age",
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "spo2",
        "temperature"
    ]


    dataset = dataset[
        [
            column
            for column in columns
            if column in dataset.columns
        ]
    ]


    # Show missingness
    print()
    print("Missing values:")
    print(
        dataset.isna().sum()
    )


    # We need all six vital signs
    required = [
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "spo2",
        "temperature"
    ]


    dataset = dataset.dropna(
        subset=required
    )


    print()
    print(
        "Complete initial-vital records:",
        len(dataset)
    )


    os.makedirs(
        "data",
        exist_ok=True
    )


    dataset.to_csv(
        OUTPUT,
        index=False
    )


    print()
    print(
        f"Dataset saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()