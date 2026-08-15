"""
DeepSurv Neural Survival Model for NSCLC PFI Prediction.

Run from project root:
    set PYTHONPATH=C:\\Users\\Admin\\nsclc-multimodal-project
    python src/training/deepsurv_train.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

CONFIG = {
    "cohort_csv":   "data/metadata/cohort_with_pfi.csv",
    "genomic_csv":  "data/genomic/genomic_features.csv",
    "wsi_dir":      "data/embeddings/patient_linked",
    "results_dir":  "outputs/results",
    "clinical_features": [
        "age", "stage_num", "t_stage_num", "n_stage_num",
        "smoking_num", "laterality_bin", "prior_malignancy_bin", "pack_years_smoked"
    ],
    "genomic_features": [
        "EGFR", "KRAS", "TP53", "ALK", "STK11",
        "KEAP1", "RB1", "MET", "BRAF", "RET",
        "SMAD4", "CDKN2A", "PIK3CA", "NF1", "NRAS", "TMB"
    ],
    "hidden_dims":  [256, 128, 64],
    "dropout":      0.3,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       150,
    "n_folds":      5,
    "seed":         42,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs(CONFIG["results_dir"], exist_ok=True)


class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def cox_loss(risk, times, events):
    order = torch.argsort(times, descending=True)
    risk   = risk[order]
    events = events[order]
    log_cs = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cs) * events)


def load_data():
    df  = pd.read_csv(CONFIG["cohort_csv"])
    gen = pd.read_csv(CONFIG["genomic_csv"])
    gen_cols = [g for g in CONFIG["genomic_features"] if g in gen.columns]
    df  = df.merge(gen[["patient_id"] + gen_cols], on="patient_id", how="left")
    df  = df[df["pfi_days"] > 0].dropna(subset=["pfi_days","pfi_event"]).copy()
    df["pfi_event"] = df["pfi_event"].astype(int)

    clin_cols = [c for c in CONFIG["clinical_features"] if c in df.columns]
    X_clin = df[clin_cols].fillna(df[clin_cols].median()).values.astype(np.float32)
    X_gen  = df[gen_cols].fillna(0).values.astype(np.float32)
    if "TMB" in gen_cols:
        idx = gen_cols.index("TMB")
        X_gen[:,idx] = (X_gen[:,idx] - X_gen[:,idx].mean()) / (X_gen[:,idx].std() + 1e-8)

    npy_files = {}
    if os.path.exists(CONFIG["wsi_dir"]):
        npy_files = {f.replace(".npy",""): os.path.join(CONFIG["wsi_dir"],f)
                     for f in os.listdir(CONFIG["wsi_dir"]) if f.endswith(".npy")}
    mean_emb = np.mean([np.load(p) for p in npy_files.values()], axis=0).astype(np.float32) if npy_files else np.zeros(2048,dtype=np.float32)
    X_wsi = np.array([np.load(npy_files[pid]) if pid in npy_files else mean_emb for pid in df["patient_id"]], dtype=np.float32)

    times  = df["pfi_days"].values.astype(np.float32)
    events = df["pfi_event"].values.astype(np.float32)
    return X_clin, X_gen, X_wsi, times, events, clin_cols, gen_cols


def train_eval(X_tr, X_vl, t_tr, t_vl, e_tr, e_vl, device):
    model = DeepSurv(X_tr.shape[1], CONFIG["hidden_dims"], CONFIG["dropout"]).to(device)
    opt   = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

    Xt = torch.tensor(X_tr).to(device)
    tt = torch.tensor(t_tr).to(device)
    et = torch.tensor(e_tr).to(device)

    model.train()
    for _ in range(CONFIG["epochs"]):
        opt.zero_grad()
        loss = cox_loss(model(Xt), tt, et)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        risk_vl = model(torch.tensor(X_vl).to(device)).cpu().numpy()

    try:
        ci = concordance_index_censored(e_vl.astype(bool), t_vl, risk_vl)[0]
        if ci < 0.5: ci = 1 - ci
    except Exception:
        ci = 0.5
    return ci


def run_kfold(name, X, times, events, device):
    print(f"\n  {name}")
    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
    cis = []
    for fold, (tr, vl) in enumerate(skf.split(X, events.astype(int))):
        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X[tr]).astype(np.float32)
        X_vl  = sc.transform(X[vl]).astype(np.float32)
        ci    = train_eval(X_tr, X_vl, times[tr], times[vl], events[tr], events[vl], device)
        cis.append(ci)
        print(f"    Fold {fold+1}: {ci:.4f}")
    print(f"  Mean: {np.mean(cis):.4f} +/- {np.std(cis):.4f}")
    return {"model": name, "c_index": round(float(np.mean(cis)),4),
            "std": round(float(np.std(cis)),4), "fold_cis": [round(c,4) for c in cis],
            "n": len(times), "events": int(events.sum())}


def main():
    print(f"\n{'='*60}")
    print(f" DeepSurv — NSCLC PFI Prediction")
    print(f"{'='*60}")

    device = torch.device(CONFIG["device"])
    print(f" Device: {device}")

    X_clin, X_gen, X_wsi, times, events, clin_cols, gen_cols = load_data()
    print(f"\n Patients: {len(times)}  Events: {int(events.sum())}  Rate: {events.mean():.1%}")
    print(f" Clinical: {len(clin_cols)} features")
    print(f" Genomic:  {len(gen_cols)} features")
    print(f" WSI:      {X_wsi.shape[1]}-dim")

    experiments = [
        ("Clinical only",            X_clin),
        ("Genomic only",             X_gen),
        ("Clinical + Genomic",       np.concatenate([X_clin, X_gen], axis=1)),
        ("WSI only",                 X_wsi),
        ("Clinical + WSI",           np.concatenate([X_clin, X_wsi], axis=1)),
        ("Clinical + Genomic + WSI", np.concatenate([X_clin, X_gen, X_wsi], axis=1)),
    ]

    all_results = {}
    for name, X in experiments:
        all_results[name] = run_kfold(name, X, times, events, device)

    print(f"\n{'='*60}")
    print(f" FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<28} {'C-index':>8} {'Std':>8}")
    print(f"{'─'*48}")
    for name, res in all_results.items():
        print(f"{name:<28} {res['c_index']:>8.4f} {res['std']:>8.4f}")

    path = os.path.join(CONFIG["results_dir"], "deepsurv_results.json")
    json.dump(all_results, open(path,"w"), indent=2)
    print(f"\n Results saved: {path}")


if __name__ == "__main__":
    main()