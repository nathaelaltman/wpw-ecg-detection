"""
age_score_confound.py -- does the deployed ensemble score track AGE among the negatives?

WPW patients are markedly younger (median ~47-48 vs ~61-62). If the model learned "young ->
higher WPW score", the detector is partly an age classifier and cross-site transfer is
suspect. Test on NEGATIVES only (where the true signal is absent, so any age-score
association is pure confound), per corpus:

  - Spearman rho( ensemble OOF score , age )   [negatives, per corpus]
  - mean ensemble score of YOUNG negatives (age <= 45) vs OLD negatives (age >= 65)
  - Mann-Whitney young-vs-old on the negative scores

rho ~ 0 and equal young/old means => confound closed (score does not encode age).

Ensemble score = frozen committee (M3+M4 rank-vote) on OOF folds 1-8. Read-only; fold 10 untouched.
Runs in seconds (all table lookups).

Run:  python age_score_confound.py
Writes: reports/metrics/age_score_confound.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
ENS  = os.path.join(ROOT, "models", "ensemble")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
ALPHA = 0.5

# ---- committee OOF scores (frozen recipe) ----
CANON = ["ecg_id","source","fold","label","proba_raw","proba_cal"]; KEY = ["ecg_id","source"]
def load_oof(fn):
    d = pd.read_csv(os.path.join(PROC, fn), dtype={"ecg_id": str}); assert list(d.columns) == CANON; return d
m3 = load_oof("m3_combined_oof.csv").rename(columns={"proba_raw": "M3"})
m4 = load_oof("m4_combined_oof.csv")[["ecg_id","source","proba_raw"]].rename(columns={"proba_raw": "M4"})
M  = m3[["ecg_id","source","fold","label","M3"]].merge(m4, on=KEY, how="inner")
ref3 = np.sort(np.load(os.path.join(ENS, "ref_scores_M3.npy")))
ref4 = np.sort(np.load(os.path.join(ENS, "ref_scores_M4.npy")))
def pct(x, ref): return np.searchsorted(ref, x, side="right") / len(ref)
M["ens"] = ALPHA*pct(M["M4"].values, ref4) + (1-ALPHA)*pct(M["M3"].values, ref3)
def is_ptb(s): return "ptb" in str(s).lower()

# ---- age per record ----
# PTB-XL
ptb_db = None
for r, _, files in os.walk(os.path.join(ROOT, "data", "raw", "ptbxl")):
    if "ptbxl_database.csv" in files: ptb_db = os.path.join(r, "ptbxl_database.csv")
P = pd.read_csv(ptb_db)[["ecg_id","age"]]; P["ecg_id"] = P["ecg_id"].astype(int).astype(str)
P = P[P.age < 300]                        # drop the >89 anonymization placeholder
age_ptb = dict(zip(P.ecg_id, P.age))
# Ningbo/CSN
N = pd.read_csv(os.path.join(PROC, "metadata_ningbo.csv"), dtype={"ecg_id": str})
age_nin = dict(zip(N.ecg_id, N.age)) if "age" in N.columns else {}

def age_of(ecg_id, source):
    return age_ptb.get(str(ecg_id), np.nan) if is_ptb(source) else age_nin.get(str(ecg_id), np.nan)

M["age"] = [age_of(r.ecg_id, r.source) for r in M.itertuples()]
M["corpus"] = np.where(M.source.map(is_ptb), "ptbxl", "csn")

def analyze(df, corpus):
    neg = df[(df.label==0) & df.age.notna()]
    rho, p = spearmanr(neg.ens, neg.age)
    young = neg[neg.age <= 45].ens; old = neg[neg.age >= 65].ens
    try: u, mp = mannwhitneyu(young, old, alternative="two-sided")
    except Exception: mp = np.nan
    return dict(corpus=corpus, n_neg=int(len(neg)),
                spearman_rho=round(float(rho),4), spearman_p=round(float(p),4),
                n_young=int(len(young)), mean_score_young=round(float(young.mean()),4),
                n_old=int(len(old)),     mean_score_old=round(float(old.mean()),4),
                mean_diff=round(float(young.mean()-old.mean()),4),
                young_vs_old_MW_p=round(float(mp),4) if np.isfinite(mp) else None)

res = [analyze(M[M.corpus=="ptbxl"], "ptbxl"),
       analyze(M[M.corpus=="csn"],   "csn"),
       analyze(M, "combined")]

print("=== ensemble score vs age among NEGATIVES ===")
for r in res:
    print("  %-9s n=%6d | Spearman rho=%+.3f (p=%.3f) | young(<=45) mean=%.4f vs old(>=65) mean=%.4f (diff=%+.4f, MW p=%s)" % (
        r["corpus"], r["n_neg"], r["spearman_rho"], r["spearman_p"],
        r["mean_score_young"], r["mean_score_old"], r["mean_diff"], r["young_vs_old_MW_p"]))
print("\n  Interpretation: rho ~ 0 and young~old means => the score does NOT encode age (confound closed).")

json.dump(dict(results=res, young_cut=45, old_cut=65,
               note="Negatives only; PTB-XL age=300 placeholder dropped; ensemble = frozen M3+M4 rank-vote."),
          open(os.path.join(MET, "age_score_confound.json"), "w"), indent=2, default=str)
print("\nWrote reports/metrics/age_score_confound.json")
