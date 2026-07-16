"""
qrs_three_delineators.py -- Q3 (extended): the critical delineator-robustness check.

On the PTB-XL WPW positives split by the DEPLOYED committee (M3+M4 rank-vote, threshold
0.9969) into detected (TP) vs missed (FN) on the OOF set (folds 1-8), measure QRS width
THREE independent ways and ask whether the "spurious narrowing of the missed WPW" survives
a change of delineator:

  (a) proxy  = _qw  (custom Pan-Tompkins detect_r + 15%-of-peak-slope span, lead II)  -- the paper's Fig C estimator
  (b) NeuroKit = QRS_ms column of data/processed/m1_features.csv (nk.ecg_peaks + ecg_delineate dwt)  -- M1's clinical delineator
  (c) 12SL   = QRS_Dur_Global of data/processed/features_marquette.csv (Marquette 12SL, PTB-XL only)  -- the commercial ground truth

For each method: median(detected) vs median(missed) + Mann-Whitney two-sided p.
Counts of records with QRS < 60 ms for (a) and (b) (implausibly narrow -> delineator failure).
Correlation (Spearman + Pearson) proxy-vs-12SL and NeuroKit-vs-12SL over the shared positives.

Interpretation the script is built to support:
  - if (b) NeuroKit and (c) 12SL ALSO show the missed WPW as narrower, the narrowing is real
    (the misses are minimal-pre-excitation WPW), not a proxy artifact;
  - if only (a) proxy narrows while (b)/(c) do not, the "narrow QRS" story is an artifact of
    the bespoke slope estimator and must be softened in Section 7.

Committee split is reconstructed with ZERO new decisions using the frozen ensemble recipe
(percentile rank vs the frozen folds-1-8 reference, threshold 0.9969). Read-only; fold 10 untouched.
Loads ~57 PTB-XL signals for (a); (b)/(c) are table lookups. Runs in seconds.

Run:  python qrs_three_delineators.py
Writes: reports/metrics/qrs_three_delineators.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import mannwhitneyu, spearmanr, pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
ENS  = os.path.join(ROOT, "models", "ensemble")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
sys.path.insert(0, SRC)
from signal_loading import load_signal, LEADS_CANONICAL

FS   = 500
iII  = list(LEADS_CANONICAL).index("II")
THR_ENS = 0.9969          # frozen committee F1-max threshold (ensemble_config.json)
ALPHA   = 0.5             # weight on M4 (ensemble_config.json)
NARROW_MS = 60.0

# ---- 0.5-40 Hz filter + Pan-Tompkins + _qw (verbatim from make_fig_qrs_tp_vs_fn.py) ----
with open(os.path.join(PROC, "filter_config.json")) as f:
    FCFG = json.load(f)["filter_FINAL"]
SOS = butter(FCFG["order"], [FCFG["low"]/(FS/2), FCFG["high"]/(FS/2)], btype="band", output="sos")
def bpf(x): return sosfiltfilt(SOS, np.asarray(x, float))

def detect_r(sig):
    d = np.diff(sig, prepend=sig[0]); w = max(1, int(0.08*FS))
    mwi = np.convolve(d*d, np.ones(w)/w, mode="same")
    pk, _ = find_peaks(mwi, distance=int(0.30*FS), height=np.mean(mwi)+0.5*np.std(mwi)); R = []
    for p in pk:
        a = max(0, p-int(0.05*FS)); b = min(len(sig), p+int(0.05*FS))
        R.append(a+int(np.argmax(np.abs(sig[a:b]))))
    return np.array(sorted(set(R)))

def _qw(sig, r):
    d = np.abs(np.diff(sig, prepend=sig[0])); w = int(0.10*FS)
    seg = d[max(0, r-w):min(len(sig), r+w)]; pk = seg.max()
    if pk <= 1e-9: return np.nan
    ab = np.where(seg > 0.15*pk)[0]
    return (ab[-1]-ab[0])/FS*1000.0 if len(ab) >= 2 else np.nan

def qrs_proxy(ecg_id, source):
    sig = bpf(load_signal(ecg_id, source)[:, iII]); R = detect_r(sig)
    if len(R) < 2: return np.nan
    qw = [x for x in (_qw(sig, r) for r in R) if np.isfinite(x)]
    return round(float(np.median(qw)), 1) if qw else np.nan

# ---- committee reconstruction (frozen recipe, ZERO new decisions) ----
CANON = ["ecg_id","source","fold","label","proba_raw","proba_cal"]; KEY = ["ecg_id","source"]
def load_oof(fn):
    d = pd.read_csv(os.path.join(PROC, fn), dtype={"ecg_id": str}); assert list(d.columns) == CANON; return d
m3 = load_oof("m3_combined_oof.csv").rename(columns={"proba_raw": "M3"})
m4 = load_oof("m4_combined_oof.csv")[["ecg_id","source","proba_raw"]].rename(columns={"proba_raw": "M4"})
M  = m3[["ecg_id","source","fold","label","M3"]].merge(m4, on=KEY, how="inner")

# percentile vs the FROZEN folds-1-8 reference scores (same as scoring new data)
ref3 = np.sort(np.load(os.path.join(ENS, "ref_scores_M3.npy")))
ref4 = np.sort(np.load(os.path.join(ENS, "ref_scores_M4.npy")))
def pct(x, ref): return np.searchsorted(ref, x, side="right") / len(ref)
M["ens"] = ALPHA*pct(M["M4"].values, ref4) + (1-ALPHA)*pct(M["M3"].values, ref3)
M["pred"] = (M["ens"].values >= THR_ENS).astype(int)

# sanity: reproduce the frozen OOF confusion (TP80/FP25/FN35)
tp = int(((M.label==1)&(M.pred==1)).sum()); fp = int(((M.label==0)&(M.pred==1)).sum()); fn = int(((M.label==1)&(M.pred==0)).sum())
print("committee OOF confusion reproduced: TP=%d FP=%d FN=%d  (expected 80/25/35)" % (tp, fp, fn))

# PTB-XL WPW only, split TP / FN
def is_ptb(s): return "ptb" in str(s).lower()
wpw_ptb = M[(M.label==1) & (M.source.map(is_ptb))].copy()
det = wpw_ptb[wpw_ptb.pred==1]; mis = wpw_ptb[wpw_ptb.pred==0]
print("PTB-XL WPW: detected=%d  missed=%d  (expected 43/14)" % (len(det), len(mis)))

# ---- (b) NeuroKit QRS_ms from m1_features.csv ----
m1 = pd.read_csv(os.path.join(PROC, "m1_features.csv"), dtype={"ecg_id": str})
qcol = "QRS_ms" if "QRS_ms" in m1.columns else next(c for c in m1.columns if c.lower()=="qrs_ms")
mkey = ["ecg_id","source"] if "source" in m1.columns else ["ecg_id"]
nk = m1[mkey+[qcol]].rename(columns={qcol: "nk_qrs"})

# ---- (c) 12SL QRS_Dur_Global from features_marquette.csv (PTB-XL) ----
fm = pd.read_csv(os.path.join(PROC, "features_marquette.csv"), dtype={"ecg_id": str})
gcol = "QRS_Dur_Global"
assert gcol in fm.columns, f"{gcol} not in features_marquette.csv"
sl = fm[["ecg_id", gcol]].rename(columns={gcol: "sl_qrs"})

def enrich(df):
    df = df.copy()
    df["proxy"] = [qrs_proxy(r.ecg_id, r.source) for r in df.itertuples()]
    df = df.merge(nk, on=mkey, how="left")
    df["ecg_id_i"] = df["ecg_id"]                       # 12SL keyed on ecg_id (PTB-XL)
    df = df.merge(sl, on="ecg_id", how="left")
    return df

det = enrich(det); mis = enrich(mis)

def summ(name, col):
    dv = det[col].dropna().values; mv = mis[col].dropna().values
    if len(dv) >= 1 and len(mv) >= 2:
        try: u, p = mannwhitneyu(dv, mv, alternative="two-sided")
        except Exception: p = np.nan
    else: p = np.nan
    return dict(method=name,
                n_det=int(len(dv)), median_det=round(float(np.median(dv)),1) if len(dv) else None,
                n_mis=int(len(mv)), median_mis=round(float(np.median(mv)),1) if len(mv) else None,
                mannwhitney_p=round(float(p),4) if np.isfinite(p) else None)

rows = [summ("(a) proxy _qw", "proxy"),
        summ("(b) NeuroKit QRS_ms", "nk_qrs"),
        summ("(c) 12SL QRS_Dur_Global", "sl_qrs")]

# <60 ms counts for (a) and (b) over ALL 57
allw = pd.concat([det, mis], ignore_index=True)
narrow = {
    "proxy_lt60":    int((allw["proxy"]  < NARROW_MS).sum()),
    "neurokit_lt60": int((allw["nk_qrs"] < NARROW_MS).sum()),
    "proxy_n_finite":    int(allw["proxy"].notna().sum()),
    "neurokit_n_finite": int(allw["nk_qrs"].notna().sum()),
}

# correlations over the shared positives (finite in both)
def corr(x, y):
    d = allw[[x, y]].dropna()
    if len(d) < 3: return dict(n=len(d), spearman=None, pearson=None)
    return dict(n=int(len(d)),
                spearman=round(float(spearmanr(d[x], d[y]).correlation), 3),
                pearson=round(float(pearsonr(d[x], d[y])[0]), 3))
cors = {"proxy_vs_12sl": corr("proxy", "sl_qrs"),
        "neurokit_vs_12sl": corr("nk_qrs", "sl_qrs"),
        "proxy_vs_neurokit": corr("proxy", "nk_qrs")}

# ---- report ----
print("\n=== QRS WIDTH: detected vs missed PTB-XL WPW, three delineators ===")
for r in rows:
    print("  %-26s det n=%2d med=%s | mis n=%2d med=%s | MW p=%s" % (
        r["method"], r["n_det"], r["median_det"], r["n_mis"], r["median_mis"], r["mannwhitney_p"]))
print("\n=== implausibly narrow QRS (<60 ms) ===")
print("  proxy   : %d / %d finite" % (narrow["proxy_lt60"], narrow["proxy_n_finite"]))
print("  NeuroKit: %d / %d finite" % (narrow["neurokit_lt60"], narrow["neurokit_n_finite"]))
print("\n=== correlation with the 12SL commercial ground truth ===")
for k, v in cors.items():
    print("  %-20s n=%s spearman=%s pearson=%s" % (k, v["n"], v["spearman"], v["pearson"]))

out = dict(committee_confusion=dict(TP=tp, FP=fp, FN=fn),
           ptbxl_split=dict(detected=len(det), missed=len(mis)),
           per_method=rows, narrow_lt60=narrow, correlations=cors,
           note="If (b) and (c) also narrow on misses, the narrowing is physiological, not a proxy artifact.")
json.dump(out, open(os.path.join(MET, "qrs_three_delineators.json"), "w"), indent=2, default=str)
print("\nWrote reports/metrics/qrs_three_delineators.json")
