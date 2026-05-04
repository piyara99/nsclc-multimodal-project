import pandas as pd
import numpy as np

luad = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', low_memory=False)
lusc = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', low_memory=False)
luad['subtype_str'] = 'LUAD'
lusc['subtype_str'] = 'LUSC'
df = pd.concat([luad, lusc], ignore_index=True)
df_dedup = df.drop_duplicates(subset='cases.submitter_id', keep='first')

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

df_dedup = df_dedup.copy()
df_dedup['age_v']     = df_dedup['demographic.age_at_index'].apply(to_float)
df_dedup['gender_v']  = df_dedup['demographic.gender'].apply(lambda x: 0.0 if str(clean(x)).lower()=='male' else (1.0 if str(clean(x)).lower()=='female' else np.nan))
df_dedup['stage_v']   = df_dedup['diagnoses.ajcc_pathologic_stage'].apply(lambda s: STAGE_MAP.get(str(clean(s)), np.nan))
df_dedup['days_v']    = df_dedup['diagnoses.days_to_last_follow_up'].apply(to_float)
df_dedup['death_v']   = df_dedup['demographic.days_to_death'].apply(lambda x: abs(to_float(x)) if not np.isnan(to_float(x)) else np.nan)
df_dedup['days_best'] = df_dedup['days_v'].combine_first(df_dedup['death_v'])

print(f"Total unique patients: {len(df_dedup)}")
print(f"\nMissing counts:")
print(f"  age missing:        {df_dedup['age_v'].isna().sum()}")
print(f"  gender missing:     {df_dedup['gender_v'].isna().sum()}")
print(f"  stage missing:      {df_dedup['stage_v'].isna().sum()}")
print(f"  days missing:       {df_dedup['days_best'].isna().sum()}")

# How many pass all required fields
mask = (df_dedup['age_v'].notna() & 
        df_dedup['gender_v'].notna() & 
        df_dedup['stage_v'].notna() & 
        df_dedup['days_best'].notna())
print(f"\nPass all clinical requirements: {mask.sum()}")

# Now check follow_up merge
luad_fu = pd.read_csv('data/clinical/raw/new_luad/follow_up.tsv', sep='\t', low_memory=False)
lusc_fu = pd.read_csv('data/clinical/raw/new_lusc/follow_up.tsv', sep='\t', low_memory=False)
df_fu = pd.concat([luad_fu, lusc_fu], ignore_index=True)
fu_patients = set(df_fu['cases.submitter_id'].unique())
clin_patients = set(df_dedup['cases.submitter_id'].unique())
both = fu_patients & clin_patients
print(f"\nPatients with follow_up data: {len(fu_patients)}")
print(f"Patients with clinical data:  {len(clin_patients)}")
print(f"Patients with BOTH:           {len(both)}")
print(f"\nPatients passing clinical AND in follow_up: {mask.sum() - (df_dedup[mask]['cases.submitter_id'].isin(fu_patients) == False).sum()}")

# Check stage missing breakdown by subtype
print(f"\nStage missing by subtype:")
for st in ['LUAD','LUSC']:
    sub = df_dedup[df_dedup['subtype_str']==st]
    print(f"  {st}: stage_missing={sub['stage_v'].isna().sum()}/{len(sub)}")