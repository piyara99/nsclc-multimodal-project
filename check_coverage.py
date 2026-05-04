import pandas as pd
import numpy as np

# Load both clinical TSVs
luad = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', low_memory=False)
lusc = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', low_memory=False)

print(f"LUAD rows: {len(luad)}, LUSC rows: {len(lusc)}")
print(f"Total rows: {len(luad) + len(lusc)}")

# Combine
df = pd.concat([luad, lusc], ignore_index=True)
df['subtype'] = df['project.project_id'].apply(
    lambda x: 'LUAD' if 'LUAD' in str(x) else 'LUSC'
)

print(f"\nCombined shape: {df.shape}")
print(f"LUAD: {(df['subtype']=='LUAD').sum()}, LUSC: {(df['subtype']=='LUSC').sum()}")

# Check key columns coverage
key_cols = [
    'demographic.age_at_index',
    'demographic.gender',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.days_to_last_follow_up',
    'diagnoses.progression_or_recurrence',
    'diagnoses.days_to_recurrence',
    'diagnoses.prior_treatment',
    'follow_ups.ecog_performance_status' if 'follow_ups.ecog_performance_status' in df.columns else None,
]

print("\n=== Feature Coverage ===")
for col in key_cols:
    if col and col in df.columns:
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"{col:50s}: {non_null}/{len(df)} ({pct:.1f}%)")

# Check recurrence label distribution
print("\n=== Recurrence Label Distribution ===")
print(df['diagnoses.progression_or_recurrence'].value_counts(dropna=False))

# Check stage distribution
print("\n=== Stage Distribution ===")
print(df['diagnoses.ajcc_pathologic_stage'].value_counts(dropna=False).head(15))