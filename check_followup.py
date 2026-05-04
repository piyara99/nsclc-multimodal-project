import pandas as pd
import numpy as np

# Load follow_up files
luad_fu = pd.read_csv('data/clinical/raw/new_luad/follow_up.tsv', sep='\t', low_memory=False)
lusc_fu = pd.read_csv('data/clinical/raw/new_lusc/follow_up.tsv', sep='\t', low_memory=False)

print(f"LUAD follow_up rows: {len(luad_fu)}")
print(f"LUSC follow_up rows: {len(lusc_fu)}")

df_fu = pd.concat([luad_fu, lusc_fu], ignore_index=True)
print(f"Combined follow_up rows: {len(df_fu)}")

# Check recurrence columns
recurrence_cols = [c for c in df_fu.columns if 'recurrence' in c.lower() or 'progression' in c.lower()]
print(f"\nRecurrence-related columns: {recurrence_cols}")

for col in recurrence_cols:
    print(f"\n{col}:")
    print(df_fu[col].value_counts(dropna=False).head(10))

# Check unique patients
print(f"\nUnique patients in follow_up: {df_fu['cases.submitter_id'].nunique()}")

# Load clinical for patient count
luad_c = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', low_memory=False)
lusc_c = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', low_memory=False)
df_c = pd.concat([luad_c, lusc_c], ignore_index=True)
print(f"Unique patients in clinical: {df_c['cases.submitter_id'].nunique()}")

# Check overlap
fu_patients = set(df_fu['cases.submitter_id'].unique())
c_patients = set(df_c['cases.submitter_id'].unique())
print(f"Patients in both: {len(fu_patients & c_patients)}")