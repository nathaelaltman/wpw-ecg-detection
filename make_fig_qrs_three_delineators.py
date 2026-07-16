"""
make_fig_qrs_three_delineators.py -- paper figure: three delineators, one set of cases.

Renders reports/figures/qrs_three_delineators.png. The SUMMARY stats (6 medians, 3 p-values,
<60 ms counts, correlations) are taken verbatim from reports/metrics/qrs_three_delineators.json
(the frozen source of truth) and printed for verification. The INDIVIDUAL points are not stored
in that JSON, so they are re-derived ONLY from the columns the original qrs_three_delineators.py
already used -- proxy _qw on lead II, NeuroKit QRS_ms from m1_features.csv, QRS_Dur_Global from
features_marquette.csv -- over the 57 committee ecg_ids. The 6-GB M3 pool is NEVER touched.

Read-only on data; writes only the PNG. fold 10 untouched.

Run:  python make_fig_qrs_three_delineators.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
ENS  = os.path.join(ROOT, "models", "ensemble")
FIG  = os.path.join(ROOT, "reports", "figures"); os.makedirs(FIG, exist_ok=True)
MET  = os.path.join(ROOT, "reports", "metrics")
sys.path.insert(0, SRC)
from signal_loading import load_signal, LEADS_CANONICAL

FS = 500; iII = list(LEADS_CANONICAL).index("II")
ALPHA = 0.5; THR_ENS = 0.9969; NARROW = 60.0
C_DET, C_MIS = "#16a34a", "#dc2626"   # green = detected, red = missed

# ---- frozen summary (source of truth) ----
S = json.load(open(os.path.join(MET, "qrs_three_delineators.json")))
PM = {r["method"]: r for r in S["per_method"]}
def pstr(p): return "p < 0.001" if (p is not None and p < 0.001) else ("p = %.3f" % p)

# ---- 0.5-40 filter + Pan-Tompkins + _qw (verbatim) ----
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
    return float(np.median(qw)) if qw else np.nan

# ---- committee split (frozen recipe, light) ----
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
M["pred"] = (M["ens"].values >= THR_ENS).astype(int)
def is_ptb(s): return "ptb" in str(s).lower()
wpw = M[(M.label==1) & (M.source.map(is_ptb))].copy()
det = wpw[wpw.pred==1]; mis = wpw[wpw.pred==0]

# ---- per-ECG values (only the columns the original script used) ----
det = det.copy(); mis = mis.copy()
for df in (det, mis):
    df["proxy"] = [qrs_proxy(r.ecg_id, r.source) for r in df.itertuples()]
m1 = pd.read_csv(os.path.join(PROC, "m1_features.csv"), dtype={"ecg_id": str})
mkey = ["ecg_id","source"] if "source" in m1.columns else ["ecg_id"]
nk = m1[mkey + ["QRS_ms"]].rename(columns={"QRS_ms": "nk"})
fm = pd.read_csv(os.path.join(PROC, "features_marquette.csv"), dtype={"ecg_id": str})
sl = fm[["ecg_id", "QRS_Dur_Global"]].rename(columns={"QRS_Dur_Global": "sl"})
def merge_vals(df):
    df = df.merge(nk, on=mkey, how="left").merge(sl, on="ecg_id", how="left")
    return df
det = merge_vals(det); mis = merge_vals(mis)

PANELS = [
    ("proxy", "(a) proxy _qw", "Delineation proxy (lead II)", True),
    ("nk",    "(b) NeuroKit QRS_ms", "NeuroKit2 delineation", True),
    ("sl",    "(c) 12SL QRS_Dur_Global", "Marquette 12SL (device, reference)", False),
]

# shared y-scale across all three panels
allvals = np.concatenate([pd.to_numeric(pd.concat([det[k], mis[k]]), errors="coerce").dropna().values
                          for k, _, _, _ in PANELS])
ymin = min(0, np.nanmin(allvals) - 8); ymax = np.nanmax(allvals) * 1.12

fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), sharey=True)
rng = np.random.default_rng(42)
for ax, (key, jkey, title, show_narrow) in zip(axes, PANELS):
    dv = pd.to_numeric(det[key], errors="coerce").dropna().values
    mv = pd.to_numeric(mis[key], errors="coerce").dropna().values
    groups = [dv, mv]; xs = [1, 2]; cols = [C_DET, C_MIS]
    # shaded <60 ms band on (a) and (b)
    if show_narrow:
        ax.axhspan(ymin, NARROW, color="#94a3b8", alpha=0.18, zorder=0)
        ax.axhline(NARROW, color="#64748b", lw=0.8, ls="--", zorder=1)
        ax.text(0.55, NARROW-3, "physiologically implausible (<60 ms)", fontsize=8.5,
                color="#475569", va="top", ha="left")
        nlt = int(((np.concatenate(groups)) < NARROW).sum())
        ax.text(1.5, ymin+5, "%d points <60 ms" % nlt, fontsize=9, color="#475569",
                ha="center", va="bottom", style="italic")
    # violin + strip
    vp = ax.violinplot(groups, positions=xs, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    for b, c in zip(vp["bodies"], cols):
        b.set_facecolor(c); b.set_alpha(0.20); b.set_edgecolor(c); b.set_linewidth(1.2)
    for x, g, c in zip(xs, groups, cols):
        jx = x + rng.uniform(-0.10, 0.10, size=len(g))
        ax.scatter(jx, g, s=32, color=c, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
        med = float(np.median(g))
        ax.plot([x-0.27, x+0.27], [med, med], color=c, lw=2.8, zorder=4)
        ax.text(x, ymax*0.995, "med %.0f" % med, ha="center", va="top", fontsize=9.5,
                color=c, fontweight="bold")
    p = PM[jkey]["mannwhitney_p"]
    ax.set_title("%s\nMann-Whitney %s" % (title, pstr(p)), fontsize=11.5)
    ax.set_xticks(xs); ax.set_xticklabels(["detected\n(n=%d)" % len(dv), "missed\n(n=%d)" % len(mv)], fontsize=10.5)
    ax.set_ylim(ymin, ymax)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.22)

axes[0].set_ylabel("QRS width (ms)", fontsize=12.5)
leg = [Patch(facecolor=C_DET, alpha=0.6, label="detected WPW (committee TP)"),
       Patch(facecolor=C_MIS, alpha=0.6, label="missed WPW (committee FN)")]
axes[2].legend(handles=leg, loc="upper right", fontsize=9.5, frameon=True)

fig.suptitle("Three delineators, one set of cases: the direction of the QRS-width difference depends on the delineator",
             fontsize=13.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = os.path.join(FIG, "qrs_three_delineators.png")
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("figure ->", OUT)

# ---- verification printout (medians/p from the frozen JSON + re-derived point medians) ----
print("\n=== verification (frozen JSON medians / p) ===")
for key, jkey, title, _ in PANELS:
    r = PM[jkey]
    dv = pd.to_numeric(det[key], errors="coerce").dropna().values
    mv = pd.to_numeric(mis[key], errors="coerce").dropna().values
    print("  %-26s det med JSON=%.1f (pts=%.1f) | mis med JSON=%.1f (pts=%.1f) | %s" % (
        title, r["median_det"], np.median(dv), r["median_mis"], np.median(mv), pstr(r["mannwhitney_p"])))
print("  <60 ms: proxy %d/57, NeuroKit %d/57, 12SL 0/57" % (
    S["narrow_lt60"]["proxy_lt60"], S["narrow_lt60"]["neurokit_lt60"]))
print("  corr vs 12SL: proxy Spearman %.2f | NeuroKit %.2f" % (
    S["correlations"]["proxy_vs_12sl"]["spearman"], S["correlations"]["neurokit_vs_12sl"]["spearman"]))
