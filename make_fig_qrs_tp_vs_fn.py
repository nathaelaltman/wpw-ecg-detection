"""
make_fig_qrs_tp_vs_fn.py -- Figure C for the WPW paper: QRS-width distribution, TP vs FN.

Replaces the heart-rate scatter (error_tp_vs_fn.png). Clean standalone distribution of
QRS-width ALONE for ensemble true positives (n=60) vs ensemble false negatives (n=27),
with jittered individual points and the two medians (101.5 vs 70.0 ms). NO heart rate on
any axis. Read-only: frozen OOF scores + raw signal, no model refit, no re-thresholding,
fold10 never touched. Machinery (band-pass, R-detection, QRS-width proxy _qw, ensemble
TP/FN definition) is copied verbatim from notebook 11_error_analysis so values match exactly.

Light: loads only the 87 TP/FN signals (~seconds). Needs the raw ECG data on disk.

Run:  python make_fig_qrs_tp_vs_fn.py
Outputs:
  reports/figures/qrs_tp_vs_fn.png              (the figure)
  reports/metrics/qrs_tp_fn_values.json          (the 87 per-ECG QRS values, reusable)
  copies both light artifacts into Github/reports/
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
FIG  = os.path.join(ROOT, "reports", "figures"); os.makedirs(FIG, exist_ok=True)
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
GH   = os.path.join(ROOT, "Github")
sys.path.insert(0, SRC)
from evaluation import f1max_threshold          # same threshold rule as every model
from signal_loading import load_signal, LEADS_CANONICAL

# ---- signal machinery (verbatim from 11_error_analysis, Bloc 0b) ----
with open(os.path.join(PROC, "filter_config.json")) as f:
    FCFG = json.load(f)["filter_FINAL"]
FS  = FCFG["fs"]
SOS = butter(FCFG["order"], [FCFG["low"]/(FS/2), FCFG["high"]/(FS/2)], btype="band", output="sos")
def bpf(x): return sosfiltfilt(SOS, np.asarray(x, float))
Lc = list(LEADS_CANONICAL); iII = Lc.index("II")

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

def qrs_med_of(ecg_id, source):
    sig = bpf(load_signal(ecg_id, source)[:, iII]); R = detect_r(sig)
    if len(R) < 2: return np.nan
    qw = [x for x in (_qw(sig, r) for r in R) if np.isfinite(x)]
    return round(float(np.median(qw)), 1) if qw else np.nan

# ---- ensemble TP / FN on the OOF set (verbatim definition) ----
CANON = ["ecg_id","source","fold","label","proba_raw","proba_cal"]; KEY = ["ecg_id","source"]
def load(fn):
    d = pd.read_csv(os.path.join(PROC, fn), dtype={"ecg_id": str}); assert list(d.columns) == CANON; return d
m3 = load("m3_combined_oof.csv")[["ecg_id","source","fold","label","proba_raw"]].rename(columns={"proba_raw":"M3"})
m4 = load("m4_combined_oof.csv")[["ecg_id","source","proba_raw"]].rename(columns={"proba_raw":"M4"})
M  = m3.merge(m4, on=KEY, how="inner")
assert int((M.label == 1).sum()) == 115, "expected 115 OOF WPW, got %d" % int((M.label == 1).sum())
for m in ["M3","M4"]:
    M[m+"_pred"] = (M[m].values >= f1max_threshold(M.label.values, M[m].values)).astype(int)
wpw = M[M.label == 1]
TP = wpw[(wpw.M3_pred == 1) & (wpw.M4_pred == 1)]   # ensemble true positives
FN = wpw[(wpw.M3_pred == 0) & (wpw.M4_pred == 0)]   # ensemble false negatives (both miss)
assert len(TP) == 60 and len(FN) == 27, "TP/FN counts drifted: %d/%d (expected 60/27)" % (len(TP), len(FN))

def qrs_series(df):
    v = [qrs_med_of(r.ecg_id, r.source) for r in df.itertuples()]
    return np.array([x for x in v if np.isfinite(x)], float)
tp_q = qrs_series(TP); fn_q = qrs_series(FN)
u, p_raw = mannwhitneyu(tp_q, fn_q, alternative="two-sided")
tp_med, fn_med = float(np.median(tp_q)), float(np.median(fn_q))
print("TP n=%d median=%.1f ms | FN n=%d median=%.1f ms | MW p_raw=%.4f" % (len(tp_q), tp_med, len(fn_q), fn_med, p_raw))
assert abs(tp_med-101.5) < 3 and abs(fn_med-70.0) < 3, "medians drifted from the notebook (101.5 / 70.0)"

json.dump(dict(tp_qrs_ms=[round(float(x),1) for x in tp_q], fn_qrs_ms=[round(float(x),1) for x in fn_q],
               tp_median=tp_med, fn_median=fn_med, mannwhitney_p_raw=round(float(p_raw),4),
               holm_p_robust_family=0.013, note="Holm p from notebook 11 robust family (qrs_med)"),
          open(os.path.join(MET, "qrs_tp_fn_values.json"), "w"), indent=2)

# ---- figure: violin + jittered points, medians, Holm p; NO heart rate ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 6))
groups = [tp_q, fn_q]; cols = ["#16a34a", "#dc2626"]; xs = [1, 2]
vp = ax.violinplot(groups, positions=xs, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
for b, c in zip(vp["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.22); b.set_edgecolor(c); b.set_linewidth(1.2)
rng = np.random.default_rng(42)
for x, g, c in zip(xs, groups, cols):
    jx = x + rng.uniform(-0.09, 0.09, size=len(g))
    ax.scatter(jx, g, s=34, color=c, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
    med = float(np.median(g))
    ax.plot([x-0.26, x+0.26], [med, med], color=c, lw=2.6, zorder=4)
    ax.text(x+0.30, med, "median %.0f ms" % med, va="center", ha="left", fontsize=11, color=c, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["True positives\n(n=%d)" % len(tp_q), "False negatives\n(n=%d)" % len(fn_q)], fontsize=12)
ax.set_ylabel("QRS width proxy (ms)", fontsize=12.5)
ax.set_title("Ensemble misses are minimal-pre-excitation WPW\n(narrow QRS: median 70 vs 101 ms)", fontsize=13)
ymax = max(groups[0].max(), groups[1].max())
ax.annotate("Mann-Whitney p = %.4f  (Holm-corrected p = 0.013)" % p_raw,
            xy=(1.5, ymax*1.02), ha="center", fontsize=11, color="#334155")
ax.set_ylim(min(tp_q.min(), fn_q.min())-8, ymax*1.12)
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
p = os.path.join(FIG, "qrs_tp_vs_fn.png"); fig.tight_layout(); fig.savefig(p, dpi=200, bbox_inches="tight")
print("figure ->", p)

# ---- copy light artifacts into the public repo ----
if os.path.isdir(GH):
    for sub, fn in [("figures","qrs_tp_vs_fn.png"), ("metrics","qrs_tp_fn_values.json")]:
        dst = os.path.join(GH, "reports", sub); os.makedirs(dst, exist_ok=True)
        src = os.path.join(ROOT, "reports", sub, fn)
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(dst, fn))
    print("copied figure + values into Github/reports/")
print("DONE.")
