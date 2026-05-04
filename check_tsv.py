import pandas as pd

print("=== LUAD clinical.tsv ===")
df = pd.read_csv('data/clinical/raw/new_luad/clinical.tsv', sep='\t', nrows=3)
print('Shape:', df.shape)
print('Columns:', list(df.columns))

print("\n=== LUSC clinical.tsv ===")
df2 = pd.read_csv('data/clinical/raw/new_lusc/clinical.tsv', sep='\t', nrows=3)
print('Shape:', df2.shape)
print('Columns:', list(df2.columns))

print("\n=== LUAD follow_up.tsv ===")
df3 = pd.read_csv('data/clinical/raw/new_luad/follow_up.tsv', sep='\t', nrows=3)
print('Shape:', df3.shape)
print('Columns:', list(df3.columns))