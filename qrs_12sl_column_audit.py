"""
qrs_12sl_column_audit.py -- resolve the 94/89 vs 140/103 contradiction.

Two analyses claim "12SL QRS width on the committee's 43 detected / 14 missed PTB-XL WPW":
  - Section 7.2 "v5" (error_analysis_committee_population.json A2): detected 94.0 / missed 89.0 / MW p=0.2647
  - qrs_three_delineators.py (QRS_Dur_Global): detected 140.0 / missed 103.0 / p~0
Only one column is the true global QRS duration. This script recomputes, for EVERY QRS-ish
column of features_marquette.csv, the per-group median (detected/missed) + Mann-Whitney p on
the exact same 43/14 committee split, so we can see which column reproduces 94/89 (likely a
MARKER column such as QRS_On_Global / QRS_Off_Global, NOT a duration) and confirm the true
QRS duration (QRS_Dur_Global).

Committee split reconstructed with the frozen recipe (identical to qrs_three_delineators.py).
Read-only; fold 10 untouched. Runs in seconds (one CSV read, no signals).

Run:  python qrs_12sl_column_audit.py
Writes: reports/metrics/qrs_12sl_column_audit.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
ENS  = os.path.join(ROOT, "models", "ensemble")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
ALPHA = 0.5; THR_ENS = 0.9969

# ---- committee reconstruction (frozen recipe) ----
CANON = ["ecg_id","source","fold","label","proba_raw","proba_cal"]; KEY = ["ecg_id","source"]
def load_oof(fn):
    d = pd.read_csv(os.path.join(PROC, fn), dtype={"ecg_id": str}); assert list(d.columns) == CANON; return d
m3 = load_oof("m3_combined_oof.csv").rename(columns={"proba_raw": "M3"})
m4 = load_oof("m4_combined_oof.csv")[["ecg_id","source","proba_raw"]].rename(columns={"proba_raw": "M4"})
M  = m3[["ecg_id","source","fold","label","M3"]].merge(m4, on=KEY, how="inner")
ref3 = np.sort(np.load(os.path.join(ENS, "ref_scores_M3.npy")))
ref4 = np.sort(np.load(os.path.join(ENS, "ref_scores_M4.npy")))
def pct(x, ref): return np.searchsorted(ref, x, side="right") / len(ref)
M["ens"]  = ALPHA*pct(M["M4"].values, ref4) + (1-ALPHA)*pct(M["M3"].values, ref3)
M["pred"] = (M["ens"].values >= THR_ENS).astype(int)
def is_ptb(s): return "ptb" in str(s).lower()
wpw = M[(M.label==1) & (M.source.map(is_ptb))].copy()
det_ids = set(wpw[wpw.pred==1].ecg_id); mis_ids = set(wpw[wpw.pred==0].ecg_id)
print("PTB-XL WPW committee split: detected=%d missed=%d (expected 43/14)" % (len(det_ids), len(mis_ids)))

# ---- load features_marquette.csv, pick all QRS-ish columns ----
fm = pd.read_csv(os.path.join(PROC, "features_marquette.csv"), dtype={"ecg_id": str})
qcols = [c for c in fm.columns if "qrs" in c.lower()]
# also include the raw QRS marker times so the marker-vs-duration confusion is visible
extra = [c for c in ["QRS_On_Global","QRS_Off_Global","Q_On_Global","Q_Off_Global"] if c in fm.columns and c not in qcols]
qcols = qcols + extra
print("QRS-ish columns found (%d): %s" % (len(qcols), ", ".join(qcols)))

fm["ecg_id"] = fm["ecg_id"].astype(str)
det = fm[fm.ecg_id.isin(det_ids)]; mis = fm[fm.ecg_id.isin(mis_ids)]

def med_mw(col):
    dv = pd.to_numeric(det[col], errors="coerce").dropna().values
    mv = pd.to_numeric(mis[col], errors="coerce").dropna().values
    p = np.nan
    if len(dv) >= 1 and len(mv) >= 2:
        try: _, p = mannwhitneyu(dv, mv, alternative="two-sided")
        except Exception: p = np.nan
    return dict(column=col, n_det=int(len(dv)), median_det=round(float(np.median(dv)),1) if len(dv) else None,
                iqr_det=[round(float(np.percentile(dv,25)),1), round(float(np.percentile(dv,75)),1)] if len(dv) else None,
                n_mis=int(len(mv)), median_mis=round(float(np.median(mv)),1) if len(mv) else None,
                iqr_mis=[round(float(np.percentile(mv,25)),1), round(float(np.percentile(mv,75)),1)] if len(mv) else None,
                mannwhitney_p=round(float(p),4) if np.isfinite(p) else None)

rows = [med_mw(c) for c in qcols]

# flag the column that reproduces the v5 A2 numbers 94/89
def matches_v5(r):
    return (r["median_det"] is not None and r["median_mis"] is not None
            and abs(r["median_det"]-94.0) <= 1.0 and abs(r["median_mis"]-89.0) <= 1.0)
hit = [r["column"] for r in rows if matches_v5(r)]

print("\n%-20s | det med [IQR]        | mis med [IQR]        | MW p   | ->v5?" % "column")
print("-"*90)
for r in rows:
    flag = "  <== reproduces v5 94/89" if matches_v5(r) else ""
    print("%-20s | %6s %-15s | %6s %-15s | %-6s |%s" % (
        r["column"], r["median_det"], str(r["iqr_det"]), r["median_mis"], str(r["iqr_mis"]),
        r["mannwhitney_p"], flag))

print("\n=== VERDICT INPUTS ===")
print("  QRS_Dur_Global is the ONLY global QRS *duration* column (ms); PROJECT_CONTEXT_2.md defines it as such.")
qd = next((r for r in rows if r["column"]=="QRS_Dur_Global"), None)
if qd:
    print("  QRS_Dur_Global: detected %s vs missed %s, MW p=%s" % (qd["median_det"], qd["median_mis"], qd["mannwhitney_p"]))
    if qd["median_mis"] is not None and qd["median_det"] is not None:
        narrower = "MISSED ARE NARROWER" if qd["median_mis"] < qd["median_det"] else "no narrowing"
        sig = "significant" if (qd["mannwhitney_p"] is not None and qd["mannwhitney_p"] < 0.05) else "NOT significant"
        print("  => on the true 12SL QRS duration: %s (%s, p=%s)" % (narrower, sig, qd["mannwhitney_p"]))
print("  Column(s) reproducing the v5 94/89: %s" % (hit if hit else "NONE (v5 read a column absent here or with different handling)"))

json.dump(dict(committee_split=dict(detected=len(det_ids), missed=len(mis_ids)),
               qrs_columns=rows, v5_target=dict(det=94.0, mis=89.0, p=0.2647),
               reproduces_v5=hit,
               true_qrs_duration_column="QRS_Dur_Global"),
          open(os.path.join(MET, "qrs_12sl_column_audit.json"), "w"), indent=2, default=str)
print("\nWrote reports/metrics/qrs_12sl_column_audit.json")
