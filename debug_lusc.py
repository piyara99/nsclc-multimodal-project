import pandas as pd
import numpy as np

luad_c = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', low_memory=False)
lusc_c = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', low_memory=False)
luad_c['subtype'] = 'LUAD'
lusc_c['subtype'] = 'LUSC'
df_c = pd.concat([luad_c, lusc_c], ignore_index=True)

df_c_dedup = df_c.drop_duplicates(subset='cases.submitter_id', keep='first').copy()

# Check subtype distribution after dedup
print("Subtype distribution after dedup:")
print(df_c_dedup['subtype'].value_counts())

# Check what project_id looks like for LUSC
lusc_rows = df_c_dedup[df_c_dedup['subtype']=='LUSC']
print(f"\nLUSC rows after dedup: {len(lusc_rows)}")
print("Sample project_ids:", lusc_rows['project.project_id'].head(5).values)

# Check stage for LUSC
print("\nLUSC stage distribution:")
print(lusc_rows['diagnoses.ajcc_pathologic_stage'].value_counts().head(10))

# The real issue: dedup keeps FIRST occurrence
# If LUAD patients appear in LUSC file too, dedup on submitter_id keeps LUAD version
# Check for overlapping patient IDs
luad_ids = set(luad_c['cases.submitter_id'].unique())
lusc_ids = set(lusc_c['cases.submitter_id'].unique())
overlap = luad_ids & lusc_ids
print(f"\nOverlapping patient IDs between LUAD and LUSC files: {len(overlap)}")
print(f"LUAD unique patients: {len(luad_ids)}")
print(f"LUSC unique patients: {len(lusc_ids)}")