import os
import json
import pandas as pd
from tqdm import tqdm

clinical_folder = "data/clinical/raw"

records = []

files = [f for f in os.listdir(clinical_folder) if f.endswith(".json")]

print("Found JSON files:", len(files))

for file in tqdm(files):

    path = os.path.join(clinical_folder, file)

    with open(path) as f:
        data = json.load(f)

    for case in data:

        record = {}

        # identifiers
        case_uuid = case.get("case_id")
        record["case_uuid"] = case_uuid

        # Extract TCGA patient barcode from filename
        patient_barcode = file.split("-")[2:5]
        patient_barcode = "TCGA-" + "-".join(patient_barcode)
        record["patient_id"] = patient_barcode

        project_id = case.get("project", {}).get("project_id")
        record["project_id"] = project_id

        if project_id == "TCGA-LUAD":
            record["subtype"] = "LUAD"
        elif project_id == "TCGA-LUSC":
            record["subtype"] = "LUSC"
        else:
            continue

        # demographics
        demo = case.get("demographic", {})
        record["gender"] = demo.get("gender")
        record["age"] = demo.get("age_at_index")

        # diagnosis
        diag_primary = None
        for d in case.get("diagnoses", []):
            if d.get("diagnosis_is_primary_disease") == "true":
                diag_primary = d
                break

        if diag_primary:
            record["stage"] = diag_primary.get("ajcc_pathologic_stage")
            record["days_followup"] = diag_primary.get("days_to_last_follow_up")

        # recurrence
        record["recurrence"] = None
        record["days_to_recurrence"] = None

        for fu in case.get("follow_ups", []):
            if fu.get("progression_or_recurrence"):
                record["recurrence"] = fu.get("progression_or_recurrence")
                record["days_to_recurrence"] = fu.get("days_to_recurrence")
                break

        # binary label for ML
        if record["recurrence"] == "Yes":
            record["recurrence_label"] = 1
        else:
            record["recurrence_label"] = 0

        records.append(record)

df = pd.DataFrame(records)

print("\nTotal patients:", len(df))

print("\nSubtype distribution:")
print(df["subtype"].value_counts())

print("\nRecurrence distribution:")
print(df["recurrence"].value_counts(dropna=False))

print("\nRecurrence label distribution:")
print(df["recurrence_label"].value_counts(dropna=False))

os.makedirs("data/metadata", exist_ok=True)

df.to_csv("data/metadata/tcga_clinical_master.csv", index=False)

print("\nSaved file: data/metadata/tcga_clinical_master.csv")