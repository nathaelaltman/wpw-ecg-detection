"""
learning_curve_run.py -- data-bottleneck learning curve for the WPW paper.

Strongest single detector M4 (frozen wavelet_env, OOF AP 0.718). Retrain on increasing
random subsets of the 115 folds-1-8 WPW positives (25/50/75/100%), keeping ALL negatives
(subsample the positive class only). At each fraction, compute out-of-fold AP on folds 1-8
using the frozen evaluation protocol (native folds, per-fold scale_pos_weight, same XGBoost
config as frozen M4). Multiple seeds per fraction (positive subsampling is noisy at these
counts). NO fold10 contact anywhere; read-only on the frozen feature matrix.

Scientific question: does OOF AP still rise from 75% to 100%, or has it plateaued?

Run:  python learning_curve_run.py
Needs the M4 feature matrix on disk (~6.7 GB). Outputs:
  reports/figures/learning_curve.png
  reports/metrics/learning_curve.json   (fraction, mean AP, std, per-seed values)
and copies both into Github/reports/.
"""
import os, json, time, shutil
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "data" / "processed" / "m4_features_wavelet_env.csv"
FEATS  = ROOT / "models" / "M4_medianbeat" / "m4_combined_wavelet_env_features.csv"
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MET = ROOT / "reports" / "metrics"; MET.mkdir(parents=True, exist_ok=True)
GH  = ROOT / "Github"

FRACTIONS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
SEEDS = list(range(8))                       # >=5 seeds for mean/spread
# frozen M4 config (08a: CFGS['d3_lr10'] + fixed params); scale_pos_weight set per training fold
XGB = dict(max_depth=3, learning_rate=0.1, n_estimators=200, min_child_weight=3,
           subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
           eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=1)

def load_folds18():
    feats = pd.read_csv(FEATS)["feature"].tolist()
    use = set(["label", "fold"] + feats)
    parts = []
    for ch in pd.read_csv(MATRIX, usecols=lambda c: c in use, chunksize=20000, low_memory=False):
        sub = ch[ch["fold"].between(1, 8)]
        if len(sub): parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    X = df[feats].to_numpy(np.float32); y = df["label"].to_numpy(int); folds = df["fold"].to_numpy(int)
    return X, y, folds, feats

def oof_ap(X, y, folds, keep):
    """Frozen OOF protocol on the retained rows: native folds 1-8, per-fold scale_pos_weight."""
    Xk, yk, fk = X[keep], y[keep], folds[keep]
    oof = np.full(len(yk), np.nan)
    for h in np.unique(fk):
        trm, vam = (fk != h), (fk == h)
        if yk[trm].sum() == 0 or yk[vam].sum() == 0:
            continue
        spw = (yk[trm] == 0).sum() / max((yk[trm] == 1).sum(), 1)
        m = XGBClassifier(scale_pos_weight=spw, **XGB).fit(Xk[trm], yk[trm])
        oof[vam] = m.predict_proba(Xk[vam])[:, 1]
    ok = ~np.isnan(oof)
    return float(average_precision_score(yk[ok], oof[ok]))

def one_task(frac, seed, X, y, folds, pos_idx, neg_idx):
    rng = np.random.default_rng(seed)
    n_pos = int(round(frac * len(pos_idx)))
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    keep = np.concatenate([sel_pos, neg_idx])
    return frac, seed, n_pos, oof_ap(X, y, folds, keep)

def main():
    t0 = time.time()
    print("loading M4 folds-1-8 subset ...", flush=True)
    X, y, folds, feats = load_folds18()
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    print("X=%s | WPW=%d | neg=%d | %d features" % (X.shape, len(pos_idx), len(neg_idx), len(feats)), flush=True)
    tasks = [(f, s) for f in FRACTIONS for s in SEEDS]
    res = Parallel(n_jobs=10)(delayed(one_task)(f, s, X, y, folds, pos_idx, neg_idx)
                              for f, s in tqdm(tasks, desc="learning-curve sweep"))
    # aggregate
    out = {}
    for frac in FRACTIONS:
        aps = [ap for (f, s, n, ap) in res if f == frac]
        npos = [n for (f, s, n, ap) in res if f == frac][0]
        out["%.2f" % frac] = dict(fraction=frac, n_wpw=int(npos),
                                  mean_AP=float(np.mean(aps)), std_AP=float(np.std(aps)),
                                  per_seed=[float(a) for a in aps])
    payload = dict(detector="M4_wavelet_env", protocol="OOF folds 1-8, native folds, per-fold spw; fold10 untouched",
                   n_wpw_total=int(len(pos_idx)), seeds=SEEDS, fractions=FRACTIONS, xgb=XGB, results=out)
    json.dump(payload, open(MET / "learning_curve.json", "w"), indent=2)

    # plateau read: is 85->100 rise within the pooled spread?
    m85, m100 = out["0.85"]["mean_AP"], out["1.00"]["mean_AP"]
    s85 = out["0.85"]["std_AP"]
    rise = m100 - m85
    verdict = ("still rising at 100%%: +%.3f from 85%% (beyond the 85%% spread %.3f)" % (rise, s85)
               if rise > s85 else
               "plateaued: 85%%->100%% change %+.3f is within the 85%% spread (%.3f)" % (rise, s85))

    # figure (light, publication style)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = [out["%.2f" % f]["fraction"] * 100 for f in FRACTIONS]
    ms = [out["%.2f" % f]["mean_AP"] for f in FRACTIONS]
    ss = [out["%.2f" % f]["std_AP"] for f in FRACTIONS]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(xs, np.array(ms) - np.array(ss), np.array(ms) + np.array(ss), color="#2563eb", alpha=0.15)
    ax.errorbar(xs, ms, yerr=ss, fmt="o-", color="#2563eb", lw=2, ms=8, capsize=4, label="M4 OOF AP (mean +/- 1 SD)")
    for x, m in zip(xs, ms): ax.annotate("%.3f" % m, (x, m), textcoords="offset points", xytext=(0, 10), fontsize=10, ha="center")
    ax.set_xlabel("Fraction of training WPW positives used (%)", fontsize=12.5)
    ax.set_ylabel("Out-of-fold Average Precision (folds 1-8)", fontsize=12.5)
    ax.set_title("Data-bottleneck learning curve (M4)\n" + verdict.split(":")[0].capitalize(), fontsize=13)
    ax.set_xticks(xs); ax.set_xticklabels([str(int(f * 100)) for f in FRACTIONS])
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=10, frameon=False)
    ax.annotate(verdict, xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=10, color="#334155")
    fig.tight_layout(); fig.savefig(FIG / "learning_curve.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    if GH.exists():
        (GH / "reports" / "figures").mkdir(parents=True, exist_ok=True); (GH / "reports" / "metrics").mkdir(parents=True, exist_ok=True)
        shutil.copy(FIG / "learning_curve.png", GH / "reports" / "figures" / "learning_curve.png")
        shutil.copy(MET / "learning_curve.json", GH / "reports" / "metrics" / "learning_curve.json")
    print("\n=== learning curve (M4) ===")
    for f in FRACTIONS:
        o = out["%.2f" % f]; print("  %3d%% (n_wpw=%3d): AP %.3f +/- %.3f" % (int(f*100), o["n_wpw"], o["mean_AP"], o["std_AP"]))
    print("VERDICT:", verdict)
    print("done in %.1f min" % ((time.time() - t0) / 60))

if __na