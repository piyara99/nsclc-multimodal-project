import pandas as pd
import numpy as np
import json
import os

print("=" * 60)
print("CLEAN LABEL DERIVATION — Phase 1")
print("=" * 60)

# ── 1. Load raw follow-up data ──────────────────────────────
luad = pd.read_csv("data/clinical/raw/new_luad/follow_up.tsv", sep="\t", low_memory=False)
lusc = pd.read_csv("data/clinical/raw/new_lusc/follow_up.tsv", sep="\t", low_memory=False)
fu = pd.concat([luad, lusc], ignore_index=True)
print(f"Raw follow-up rows: {len(fu)}")

# ── 2. Extract patient ID ───────────────────────────────────
fu["patient_id"] = fu["follow_ups.submitter_id"].str.extract(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)")

# ── 3. Convert timing fields ────────────────────────────────
fu["days_to_recurrence"]  = pd.to_numeric(fu["follow_ups.days_to_recurrence"],  errors="coerce")
fu["days_to_progression"] = pd.to_numeric(fu["follow_ups.days_to_progression"], errors="coerce")
fu["days_to_follow_up"]   = pd.to_numeric(fu["follow_ups.days_to_follow_up"],   errors="coerce")

# ── 4. Derive per-patient recurrence signal ─────────────────
# Rule: patient is recurrence=1 if ANY follow-up row has
#       progression_or_recurrence == Yes
# Label quality tier:
#   GOLD   - imaging/biopsy confirmed
#   SILVER - Yes with numeric days_to_recurrence
#   BRONZE - Yes, no timing data
#   NEG    - no recurrence signal in any follow-up row

records = []
for pid, grp in fu.groupby("patient_id"):
    yes_rows = grp[grp["follow_ups.progression_or_recurrence"] == "Yes"]

    if len(yes_rows) == 0:
        max_followup = grp["days_to_follow_up"].max()
        records.append({
            "patient_id": pid,
            "clean_label": 0,
            "label_tier": "NEG",
            "days_to_event": max_followup if pd.notna(max_followup) else np.nan,
            "event_occurred": 0,
            "imaging_confirmed": 0,
        })
    else:
        imaging_rows = yes_rows[
            yes_rows["follow_ups.evidence_of_recurrence_type"].isin([
                "Convincing Image Source",
                "Biopsy with Histologic Confirmation"
            ])
        ]

        # Best timing: recurrence days first, then progression, then follow-up
        best_days = yes_rows["days_to_recurrence"].min()
        if pd.isna(best_days):
            best_days = yes_rows["days_to_progression"].min()
        if pd.isna(best_days):
            best_days = yes_rows["days_to_follow_up"].min()

        if len(imaging_rows) > 0:
            tier = "GOLD"
        elif pd.notna(yes_rows["days_to_recurrence"].min()):
            tier = "SILVER"
        else:
            tier = "BRONZE"

        records.append({
            "patient_id": pid,
            "clean_label": 1,
            "label_tier": tier,
            "days_to_event": best_days,
            "event_occurred": 1,
            "imaging_confirmed": 1 if len(imaging_rows) > 0 else 0,
        })

label_df = pd.DataFrame(records)
print()
print("Label tier distribution:")
print(label_df["label_tier"].value_counts())
print()
print(f"Total patients labelled: {len(label_df)}")
print(f"Recurrence=1 : {label_df['clean_label'].sum()}")
print(f"Recurrence=0 : {(label_df['clean_label'] == 0).sum()}")

# ── 5. Merge with patient manifest ─────────────────────────
manifest = pd.read_csv("data/patient_linked/patient_slide_manifest.csv")
print(f"\nManifest patients: {len(manifest)}")

merged = manifest.merge(label_df, on="patient_id", how="inner")
print(f"After merge with manifest: {len(merged)}")
print(f"  Recurrence=1 : {merged['clean_label'].sum()}")
print(f"  Recurrence=0 : {(merged['clean_label'] == 0).sum()}")

# ── 6. Quality breakdown in merged cohort ──────────────────
print()
print("Label tiers in final cohort:")
print(merged["label_tier"].value_counts())
print()
print(f"Patients with timing data (days_to_event non-null): "
      f"{merged['days_to_event'].notna().sum()} / {len(merged)}")

# ── 7. Save outputs ─────────────────────────────────────────
os.makedirs("data/metadata", exist_ok=True)

# Full clean cohort — all tiers
merged.to_csv("data/metadata/clean_labels_full.csv", index=False)
print(f"\nSaved: data/metadata/clean_labels_full.csv  ({len(merged)} patients)")

# High-quality subset — GOLD + SILVER + NEG only (no BRONZE)
hq = merged[merged["label_tier"].isin(["GOLD", "SILVER", "NEG"])].copy()
hq.to_csv("data/metadata/clean_labels_hq.csv", index=False)
print(f"Saved: data/metadata/clean_labels_hq.csv   ({len(hq)} patients)")

# Summary JSON
summary = {
    "total_patients": len(merged),
    "recurrence_1": int(merged["clean_label"].sum()),
    "recurrence_0": int((merged["clean_label"] == 0).sum()),
    "label_tiers": merged["label_tier"].value_counts().to_dict(),
    "with_timing": int(merged["days_to_event"].notna().sum()),
    "imaging_confirmed": int(merged["imaging_confirmed"].sum()),
    "hq_patients": len(hq),
}
with open("data/metadata/clean_label_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: data/metadata/clean_label_summary.json")

print()
print("=" * 60)
print("Phase 1 Track A — COMPLETE")
print("=" * 60)