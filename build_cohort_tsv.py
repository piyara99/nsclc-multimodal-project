"""
Build expanded TCGA cohort from TSV files.
Fixed: subtype encoding bug resolved by mapping AFTER merge using subtype_str.
Run from project root: python build_cohort_tsv.py
"""
import pandas as pd
import numpy as np
import os, shutil

print("Loading clinical TSV files...")
luad_c = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', low_memory=False)
lusc_c = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', low_memory=False)
luad_c['subtype_str'] = 'LUAD'
lusc_c['subtype_str'] = 'LUSC'
df_c = pd.concat([luad_c, lusc_c], ignore_index=True)
print(f"  Clinical: {df_c['cases.submitter_id'].nunique()} unique patients")

print("Loading follow_up TSV files...")
luad_fu = pd.read_csv('data/clinical/raw/new_luad/follow_up.tsv', sep='\t', low_memory=False)
lusc_fu = pd.read_csv('data/clinical/raw/new_lusc/follow_up.tsv', sep='\t', low_memory=False)
df_fu = pd.concat([luad_fu, lusc_fu], ignore_index=True)
print(f"  Follow-up: {df_fu['cases.submitter_id'].nunique()} unique patients")

def clean(v):
    s = str(v).strip()
    return np.nan if s in ("'--",'--','nan','None','') else s

def to_float(v):
    try: return float(clean(v))
    except: return np.nan

STAGE_MAP = {
    'Stage I':1,'Stage IA':1,'Stage IB':1,
    'Stage II':2,'Stage IIA':2,'Stage IIB':2,
    'Stage III':3,'Stage IIIA':3,'Stage IIIB':3,
    'Stage IV':4,'Stage IVA':4,'Stage IVB':4,
}

def map_stage(s):
    s = clean(s)
    if s is None or (isinstance(s,float) and np.isnan(s)): return np.nan
    return STAGE_MAP.get(str(s), np.nan)

def map_gender(g):
    g = clean(g)
    if g is None: return np.nan
    g = str(g).lower()
    if g=='male': return 0.0
    if g=='female': return 1.0
    return np.nan

def map_prior(v):
    v = clean(v)
    if v is None: return np.nan
    v = str(v).lower()
    if v=='yes': return 1.0
    if v=='no': return 0.0
    return np.nan

# Recurrence labels from follow_up
print("\nExtracting recurrence labels...")
df_fu['recurrence_label'] = df_fu['follow_ups.progression_or_recurrence'].apply(
    lambda x: 1 if str(x).strip().lower()=='yes' else 0)
df_fu['days_fu'] = df_fu['follow_ups.days_to_follow_up'].apply(to_float)

rec = df_fu.groupby('cases.submitter_id').agg(
    recurrence_label=('recurrence_label','max'),
    days_followup_fu=('days_fu', lambda x: max([v for v in x if not np.isnan(v)], default=np.nan))
).reset_index()
print(f"  Recurrence=1: {(rec['recurrence_label']==1).sum()}, Recurrence=0: {(rec['recurrence_label']==0).sum()}")

# Clinical features — keep subtype_str, do NOT create numeric subtype yet
print("\nExtracting clinical features...")
df_c_dedup = df_c.drop_duplicates(subset='cases.submitter_id', keep='first').copy()

clin = pd.DataFrame()
clin['patient_id']      = df_c_dedup['cases.submitter_id'].values
clin['subtype_str']     = df_c_dedup['subtype_str'].values          # keep as string
clin['age']             = df_c_dedup['demographic.age_at_index'].apply(to_float)
clin['gender']          = df_c_dedup['demographic.gender'].apply(map_gender)
clin['stage_numeric']   = df_c_dedup['diagnoses.ajcc_pathologic_stage'].apply(map_stage)
clin['prior_treatment'] = df_c_dedup['diagnoses.prior_treatment'].apply(map_prior)
clin['days_clin']       = df_c_dedup['diagnoses.days_to_last_follow_up'].apply(to_float)
clin['days_death']      = df_c_dedup['demographic.days_to_death'].apply(
    lambda x: abs(to_float(x)) if not np.isnan(to_float(x)) else np.nan)

print(f"  LUAD: {(clin['subtype_str']=='LUAD').sum()}, LUSC: {(clin['subtype_str']=='LUSC').sum()}")

# Merge
print("\nMerging...")
merged = clin.merge(rec, left_on='patient_id', right_on='cases.submitter_id', how='inner')
merged = merged.drop(columns=['cases.submitter_id'])
merged['days_followup'] = merged['days_clin'].combine_first(
    merged['days_followup_fu']).combine_first(merged['days_death'])
merged = merged.drop(columns=['days_clin','days_followup_fu','days_death'])

print(f"  After merge: {len(merged)}")
print(f"  LUAD: {(merged['subtype_str']=='LUAD').sum()}, LUSC: {(merged['subtype_str']=='LUSC').sum()}")

# Drop missing required (stage and days are the main droppers)
required = ['age','gender','stage_numeric','days_followup','recurrence_label']
before = len(merged)
merged = merged.dropna(subset=required)
print(f"\n  After dropping missing: {len(merged)} (dropped {before-len(merged)})")
print(f"  LUAD: {(merged['subtype_str']=='LUAD').sum()}, LUSC: {(merged['subtype_str']=='LUSC').sum()}")

# NOW map subtype to numeric AFTER filtering
merged['subtype'] = merged['subtype_str'].map({'LUAD': 0.0, 'LUSC': 1.0})

# Prior treatment coverage
pt_cov = merged['prior_treatment'].notna().mean()
print(f"\n  prior_treatment coverage: {pt_cov*100:.1f}%")
if pt_cov < 0.70:
    print("  Dropping prior_treatment")
    merged = merged.drop(columns=['prior_treatment'])
else:
    print("  Keeping prior_treatment")

# Dedup
before = len(merged)
merged = merged.drop_duplicates(subset='patient_id', keep='first')
print(f"  After dedup: {len(merged)} (dropped {before-len(merged)})")

# Final columns
final_cols = ['patient_id','age','gender','subtype','stage_numeric','days_followup']
if 'prior_treatment' in merged.columns:
    final_cols.append('prior_treatment')
final_cols.append('recurrence_label')
merged = merged[final_cols]

print(f"\n{'='*50}")
print(f" FINAL COHORT SUMMARY")
print(f"{'='*50}")
print(f"  Total patients : {len(merged)}")
print(f"  Recurrence=1   : {(merged['recurrence_label']==1).sum()} ({100*(merged['recurrence_label']==1).mean():.1f}%)")
print(f"  Recurrence=0   : {(merged['recurrence_label']==0).sum()} ({100*(merged['recurrence_label']==0).mean():.1f}%)")
print(f"  LUAD           : {(merged['subtype']==0.0).sum()}")
print(f"  LUSC           : {(merged['subtype']==1.0).sum()}")
print(f"  Age mean±std   : {merged['age'].mean():.1f} ± {merged['age'].std():.1f}")
print(f"  Stage dist:")
for s,n in merged['stage_numeric'].value_counts().sort_index().items():
    print(f"    Stage {int(s)}: {n}")
print(f"  Columns: {list(merged.columns)}")

out = 'data/metadata/tcga_clinical_master_deduped.csv'
old = 'data/metadata/tcga_clinical_master_deduped_OLD_n155.csv'
if os.path.exists(out) and not os.path.exists(old):
    shutil.copy(out, old)
    print(f"\n  Backed up → {old}")
merged.to_csv(out, index=False)
print(f"  Saved → {out}")
print(f"\nDone. Next: python src/phase1_stats.py")