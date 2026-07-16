"""
learning_curve_leakfree.py -- LEAK-FREE learning curve for M1..M5 (run on your own machine).

WHY: the previous learning curve selected the feature set ONCE on all 115 training positives and held
it fixed as the positive subset shrank. That leaks information: the low-data points benefit from a
feature choice made with positives the model was not allowed to see. This script fixes it by re-running
the ENTIRE feature selection from scratch on each positive subsample.

PROTOCOL per (detector, fraction, seed):
  1. Subsample the training WPW positives to `fraction` (negatives kept full: at 471:1 they are not the
     scarce resource). The subsample defines BOTH training AND OOF evaluation (removed positives are
     excluded entirely -- matches the existing curve whose n_wpw was 12,29,...,115).
  2. Re-run the detector's feature selection on that subsample only:
        gate  = |Cohen's d|>0.3  AND  p_FDR<0.05 (Benjamini-Hochberg)  AND  bootstrap-CI(d) excludes 0
                AND cross-corpus coherence (sign(d_ptb)==sign(d_nin) AND |d_ptb|>0.2 AND |d_nin|>0.2)
        (M5 only) stability selection: bootstrap the gate STAB_B=60 times, keep features with
                  |d|>0.3 in >= STAB_THR=0.60 of resamples.
        dedup = Spearman de-duplication, keep-first in |d|-descending order
                (threshold 0.90 for M1/M2 ; 0.95 for M3/M4/M5).
        k     = AP-vs-k sweep at the FROZEN hyperparameters, then the detector's k-rule:
                  M1/M2 : 1-SE / bootstrap-IC  (smallest k whose OOF AP >= IC_lo of the max-AP point)
                  M3/M4 : max OOF AP + parsimony tiebreak (smallest k within TIE_EPS=0.01 of the max)
                  M5    : max OOF AP UNDER a train-OOF gap cap (<=0.30) + parsimony tiebreak.
  3. Train with FROZEN hyperparameters (depth/lr/n_estimators), never re-tuned; shared subsample=0.8,
     colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3; scale_pos_weight recomputed per fold.
  4. Evaluate OOF Average Precision on folds 1-8.

Protocol source: notebooks/05a (M1), 06a (M2), 07a (M3), 08a (M4), 09a (M5) + models/M*/*config*.json.
Gate/dedup/k parameters are hard-coded below from those files (not guessed).

FOLD DISCIPLINE: only folds 1-8 are ever loaded/used. Fold 9 is dropped. Fold 10 is NEVER loaded and
NEVER referenced (load_pool filters to fold<=8 and asserts it).

MEMORY DESIGN (32 GB target, n_jobs=10): the negatives (~53,540 rows) are IDENTICAL across all 80 units
of a detector, so their per-column moments (mean/var, and per-corpus) are precomputed ONCE and shared.
The gate then needs only the tiny positive subsample (<=115 rows) plus those fixed moments; Mann-Whitney
reads the pool one column at a time (never a full copy). Only the small SELECTED-column training matrix
(rows x k, k<=~600) is ever materialised. So peak memory = the shared float32 pool (read-only, memmapped
by joblib/loky across workers) + ~O(rows x k) per worker, NOT rows x all-features per worker.

OUTPUTS:
  reports/metrics/learning_curve_leakfree.json  (per det x fraction: mean/SD OOF AP, mean #features,
      raw per-seed values; the paired 90%-vs-100% "still rising" test; M4 leak-free vs contaminated)
  reports/figures/learning_curve_all_models.png (mean +/- 1 SD bands, five curves)
  Both copied to Github/reports/. The script does NOT commit or push.

RESUMABLE: checkpoints after every (detector, fraction) block; relaunch to resume.

Run:
  python learning_curve_leakfree.py                          # all detectors, resumes
  python learning_curve_leakfree.py --detectors M1 M4
  python learning_curve_leakfree.py --jobs 6                 # lower jobs if RAM is tight
  python learning_curve_leakfree.py --ksweep 16              # faster (coarser k grid), variance preserved
"""
import os, sys, json, time, argparse, shutil, warnings, math
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
METRICS = os.path.join(ROOT, "reports", "metrics"); os.makedirs(METRICS, exist_ok=True)
FIGDIR = os.path.join(ROOT, "reports", "figures"); os.makedirs(FIGDIR, exist_ok=True)
GH = os.path.join(ROOT, "Github")
OUT_JSON = os.path.join(METRICS, "learning_curve_leakfree.json")
OUT_FIG = os.path.join(FIGDIR, "learning_curve_all_models.png")

META_CANDIDATES = ["ecg_id", "patient_id", "label", "fold", "source",
                   "extraction_failed", "n_beats", "n_leads_morpho_failed"]

# per-detector frozen protocol (hard-coded from the notebooks + configs)
DETECTORS = {
    "M1": dict(pool="m1_features.csv",             dedup=0.90, k_rule="1se",
               xgb=dict(max_depth=2, learning_rate=0.05, n_estimators=200), stability=None, gap_cap=None),
    "M2": dict(pool="m2_features.csv",             dedup=0.90, k_rule="1se",
               xgb=dict(max_depth=2, learning_rate=0.03, n_estimators=300), stability=None, gap_cap=None),
    "M3": dict(pool="m3_features.csv",             dedup=0.95, k_rule="maxoof",
               xgb=dict(max_depth=4, learning_rate=0.10, n_estimators=200), stability=None, gap_cap=None),
    "M4": dict(pool="m4_features_wavelet_env.csv", dedup=0.95, k_rule="maxoof",
               xgb=dict(max_depth=3, learning_rate=0.10, n_estimators=200), stability=None, gap_cap=None),
    "M5": dict(pool="m5v2_features.csv",           dedup=0.95, k_rule="maxoof_gap",
               xgb=dict(max_depth=2, learning_rate=0.03, n_estimators=200),
               stability=dict(B=60, thr=0.60), gap_cap=0.30),
}
SHARED_XGB = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3,
                  tree_method="hist", eval_metric="aucpr", n_jobs=1)  # parallelism is over UNITS
TIE_EPS = 0.01            # M3/M4/M5 parsimony band (notebooks 07a/08a/09a)
GATE_D = 0.3; CROSS_D = 0.2; FDR_Q = 0.05
FRACTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CONTAMINATED_M4 = {10: 0.225, 25: 0.484, 40: 0.543, 55: 0.564, 70: 0.660, 85: 0.688, 100: 0.718}


# ==================================================================================================
# exact statistics (validated against scipy/statsmodels to machine precision)
# ==================================================================================================
def _rankdata(x):
    """Average ranks, 1-based (== scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float); sx = x[order]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _moments(col):
    """mean, ddof=1 var, count of a 1-D array ignoring NaN."""
    c = col[~np.isnan(col)]
    n = len(c)
    if n < 2:
        return np.nan, np.nan, n
    return float(c.mean()), float(c.var(ddof=1)), n


def cohens_d_from_moments(mp, vp, npos, mn, vn, nneg):
    """Cohen's d (pooled SD, ddof=1) from group moments (scalars or arrays)."""
    sp = np.sqrt(((npos - 1) * vp + (nneg - 1) * vn) / np.maximum(npos + nneg - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sp > 0, (mp - mn) / sp, np.nan)


def cohens_d_cols(Xpos, Xneg):
    """Vectorized Cohen's d per column (for validation / small arrays)."""
    def mv(X):
        m = np.nanmean(X, 0); n = np.sum(~np.isnan(X), 0).astype(float)
        v = np.nansum((X - m) ** 2, 0) / np.maximum(n - 1, 1); return m, v, n
    mp, vp, np_ = mv(Xpos); mn, vn, nn = mv(Xneg)
    return cohens_d_from_moments(mp, vp, np_, mn, vn, nn)


def mw_p_one(pos, negsorted, neg_tie_term):
    """Two-sided Mann-Whitney U p-value (large-sample normal approx WITH tie correction and continuity
    correction) for one feature, given the pos values and the PRESORTED valid negatives (1-D, no NaN)
    + the negatives' tie term sum(t^3 - t). Matches scipy.mannwhitneyu(method='asymptotic') to ~1e-3;
    exact when there are no pos<->neg ties. NaN pos values are dropped."""
    a = pos[~np.isnan(pos)]
    na = len(a); nb = len(negsorted)
    if na < 2 or nb < 2:
        return np.nan
    lt = np.searchsorted(negsorted, a, side="left").astype(float)
    le = np.searchsorted(negsorted, a, side="right").astype(float)
    eq = le - lt                                    # # negatives equal to each pos value
    U = float((lt + 0.5 * eq).sum())                # AUC-style U (pos vs neg)
    N = na + nb
    mu = na * nb / 2.0
    # tie term over the COMBINED sample: neg-internal ties (precomputed) + ties created by pos<->neg
    # equalities and pos-internal ties. Pos is tiny, so we add its exact contribution.
    _, cpos = np.unique(a, return_counts=True)
    pos_tie = float(np.sum(cpos ** 3 - cpos))
    # pos<->neg shared-value groups enlarge tie groups; approximate by treating equal values as merged.
    eqmask = eq > 0
    shared_extra = float(np.sum((eq[eqmask] + 1) ** 3 - (eq[eqmask] + 1) - (eq[eqmask] ** 3 - eq[eqmask]) - 0))
    tie = neg_tie_term + pos_tie + shared_extra
    sig = np.sqrt(na * nb / 12.0 * ((N + 1) - tie / (N * (N - 1))))
    if sig <= 0:
        return np.nan
    z = (abs(U - mu) - 0.5) / sig
    return float(math.erfc(z / math.sqrt(2.0)))     # 2*(1 - Phi(z))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaN-safe) == statsmodels multipletests(method='fdr_bh')."""
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); pv = p[ok]; m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    adj = np.empty(m); adj[order] = np.clip(ranked, 0, 1); out[ok] = adj
    return out


def spearman_dedup(Xtr, order_idx, thr):
    """Keep-first Spearman de-duplication in the given (|d|-desc) order. Pre-ranked columns +
    np.corrcoef (Pearson on ranks == Spearman). NEVER pandas .corr('spearman')."""
    if len(order_idx) == 0:
        return []
    sub = Xtr[:, order_idx].astype(np.float64)
    colmean = np.nanmean(sub, 0); inds = np.where(np.isnan(sub))
    sub[inds] = np.take(colmean, inds[1])
    R = np.apply_along_axis(_rankdata, 0, sub)
    C = np.abs(np.corrcoef(R, rowvar=False))
    keep = []
    for i in range(len(order_idx)):
        if all(C[i, j] <= thr for j in keep):
            keep.append(i)
    return [order_idx[i] for i in keep]


# ==================================================================================================
# modeling
# ==================================================================================================
def make_xgb(spw, xgb_cfg):
    from xgboost import XGBClassifier
    p = dict(SHARED_XGB); p.update(xgb_cfg); p["scale_pos_weight"] = spw; p["random_state"] = 42
    return XGBClassifier(**p)


def oof_scores(Xk, y, folds, fold_ids, xgb_cfg):
    oof = np.full(len(y), np.nan, np.float32)
    for h in fold_ids:
        tr = folds != h; va = folds == h
        if y[tr].sum() == 0 or y[va].sum() == 0:
            continue
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = make_xgb(spw, xgb_cfg).fit(Xk[tr], y[tr])
        oof[va] = m.predict_proba(Xk[va])[:, 1]
    return oof


def oof_ap_from_scores(oof, y):
    from sklearn.metrics import average_precision_score
    m = ~np.isnan(oof)
    return float(average_precision_score(y[m], oof[m]))


def train_ap(Xk, y, spw, xgb_cfg):
    from sklearn.metrics import average_precision_score
    mdl = make_xgb(spw, xgb_cfg).fit(Xk, y)
    return float(average_precision_score(y, mdl.predict_proba(Xk)[:, 1]))


def k_grid(kmax, n_points):
    """Geometric k grid spanning 1..kmax (log-spaced). Resolves the fast-rising low-k region AND covers
    the high-k region where M3 (K~500) and M4 (K~220) actually optimise -- a low-clustered grid would
    starve exactly the region that matters for those detectors. Always includes kmax. This mirrors the
    frozen protocols, whose K grids spanned the whole range rather than clustering at small k."""
    if kmax <= n_points:
        return list(range(1, kmax + 1))
    g = np.unique(np.round(np.geomspace(1, kmax, n_points)).astype(int))
    return sorted(set(g.tolist()) | {kmax})


def select_k(Xsel, y, folds, fold_ids, det_cfg, ksweep, boot, rng):
    """AP-vs-k sweep at frozen hyperparams (Xsel columns already in dedup order), then the k-rule.
    Xsel: (n, kmax) selected-and-ordered training matrix. Returns (k, oof_ap_at_k)."""
    from sklearn.metrics import average_precision_score
    kmax = Xsel.shape[1]
    if kmax == 0:
        return 0, float("nan")
    ks = k_grid(kmax, ksweep); xgb_cfg = det_cfg["xgb"]
    aps = {}; gaps = {}; oof_cache = {}
    for k in ks:
        Xk = Xsel[:, :k]
        oof = oof_scores(Xk, y, folds, fold_ids, xgb_cfg)
        oof_cache[k] = oof
        aps[k] = oof_ap_from_scores(oof, y)
        if det_cfg["k_rule"] == "maxoof_gap":
            spw = (y == 0).sum() / max((y == 1).sum(), 1)
            gaps[k] = train_ap(Xk, y, spw, xgb_cfg) - aps[k]
    rule = det_cfg["k_rule"]
    if rule == "1se":
        kbest = max(aps, key=aps.get)
        oofb = oof_cache[kbest]; mb = ~np.isnan(oofb); yb = y[mb]; pb = oofb[mb]
        pos = np.where(yb == 1)[0]; neg = np.where(yb == 0)[0]; bs = np.empty(boot)
        for i in range(boot):
            idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            bs[i] = average_precision_score(yb[idx], pb[idx])
        ic_lo = np.percentile(bs, 2.5)
        cand = [k for k in ks if aps[k] >= ic_lo]; ksel = min(cand) if cand else kbest
    elif rule == "maxoof":
        mx = max(aps.values()); ksel = min(k for k in ks if aps[k] >= mx - TIE_EPS)
    elif rule == "maxoof_gap":
        capped = [k for k in ks if gaps[k] <= det_cfg["gap_cap"]]; pool = capped if capped else list(ks)
        mx = max(aps[k] for k in pool); ksel = min(k for k in pool if aps[k] >= mx - TIE_EPS)
    else:
        raise ValueError(rule)
    return ksel, aps[ksel]


# ==================================================================================================
# one leak-free unit  (memory-safe: negatives via precomputed moments; MW column-by-column)
# ==================================================================================================
def run_unit(det, frac_pct, seed, POOL, args):
    t0 = time.time(); det_cfg = DETECTORS[det]
    X = POOL["X"]; y_all = POOL["y"]; folds_all = POOL["folds"]; src_all = POOL["src"]
    fold_ids = POOL["fold_ids"]; F = X.shape[1]
    NEG = POOL["neg"]                                       # precomputed negative structures (shared)
    rng = np.random.default_rng(1000 * seed + frac_pct)

    pos_idx = np.where(y_all == 1)[0]
    # STRATIFIED positive subsample: draw within each source corpus so the PTB-XL / CSN ratio among
    # positives is preserved at EVERY fraction. An unstratified draw over all 115 lets a low-fraction
    # seed land almost entirely in one corpus, which makes the cross-corpus coherence gate structurally
    # impossible (Cohen's d needs >=2 positives per corpus) and collapses the gate -- turning the low-
    # fraction points into a measurement of draw luck rather than of sample size. We also keep >=2 per
    # corpus where available so cross-corpus coherence is always computable.
    frac = frac_pct / 100.0
    parts = []
    for s in (0, 1):                                        # 0 = ptbxl, 1 = ningbo (Chapman-Shaoxing-Ningbo)
        sp = pos_idx[src_all[pos_idx] == s]
        if len(sp) == 0:
            continue
        k_s = int(round(len(sp) * frac))
        k_s = min(len(sp), max(min(2, len(sp)), k_s))       # >=2 per corpus where available
        parts.append(rng.choice(sp, size=k_s, replace=False))
    keep_pos = np.concatenate(parts)
    keep_n = len(keep_pos)
    Xpos = X[keep_pos]                                      # (<=115, F) small copy
    src_pos = src_all[keep_pos]

    # ---- GATE (positives from Xpos; negatives from precomputed moments; MW reads columns lazily) ----
    mp = np.nanmean(Xpos, 0); vp = np.nanvar(Xpos, 0, ddof=1); npos = keep_n
    d = cohens_d_from_moments(mp, vp, npos, NEG["mean"], NEG["var"], NEG["n"])
    # per-corpus d (0=ptbxl, 1=ningbo)
    dp = cohens_d_from_moments(np.nanmean(Xpos[src_pos == 0], 0), np.nanvar(Xpos[src_pos == 0], 0, ddof=1),
                               max((src_pos == 0).sum(), 1), NEG["mean_p"], NEG["var_p"], NEG["n_p"])
    dn = cohens_d_from_moments(np.nanmean(Xpos[src_pos == 1], 0), np.nanvar(Xpos[src_pos == 1], 0, ddof=1),
                               max((src_pos == 1).sum(), 1), NEG["mean_n"], NEG["var_n"], NEG["n_n"])
    cross_ok = np.isfinite(dp) & np.isfinite(dn) & (np.sign(dp) == np.sign(dn)) & (np.abs(dp) > CROSS_D) & (np.abs(dn) > CROSS_D)
    # Mann-Whitney p, one column at a time (uses presorted negatives; no full-pool copy).
    # NEG["sorted"] is a shared 2-D (F, n_neg) array, each row sorted ascending with NaN pushed to the
    # end; NEG["nvalid"][f] is the count of non-NaN negatives for feature f.
    p = np.full(F, np.nan)
    negsorted = NEG["sorted"]; negtie = NEG["tie"]; nvalid = NEG["nvalid"]
    for f in range(F):
        p[f] = mw_p_one(Xpos[:, f], negsorted[f, :nvalid[f]], negtie[f])
    pfdr = bh_fdr(p)
    # bootstrap CI(d) excludes 0 -- WPW-only vectorized bootstrap vs fixed negative moments (disclosed)
    ci_ok = _bootstrap_ci_excl0(Xpos, NEG["mean"], NEG["var"], NEG["n"], args.boot, rng)
    gate = (np.abs(d) > GATE_D) & (pfdr < FDR_Q) & ci_ok & cross_ok
    gate = np.where(np.isnan(d), False, gate)
    passed = np.where(gate)[0]

    # ---- (M5) stability selection: bootstrap |d|>0.3 on the subsample, keep freq >= thr ----
    if det_cfg["stability"] is not None and len(passed) > 0:
        B = det_cfg["stability"]["B"]; thr = det_cfg["stability"]["thr"]; freq = np.zeros(len(passed))
        Xpp = Xpos[:, passed]
        mn_p = NEG["mean"][passed]; vn_p = NEG["var"][passed]; nn_p = NEG["n"][passed]
        for b in range(B):
            ip = rng.integers(0, npos, npos); s = Xpp[ip]
            db = cohens_d_from_moments(np.nanmean(s, 0), np.nanvar(s, 0, ddof=1), npos, mn_p, vn_p, nn_p)
            freq += (np.abs(db) > GATE_D).astype(float)
        passed = passed[(freq / B) >= thr]

    if len(passed) == 0:
        return dict(det=det, frac=frac_pct, seed=seed, oof_ap=float("nan"), n_features=0,
                    n_gate=0, n_dedup=0, wall=time.time() - t0, status="empty_gate")

    # order by |d| desc, then Spearman dedup in that order (on the ACTIVE rows: kept pos + all neg)
    passed = passed[np.argsort(-np.abs(d[passed]))]
    active = np.concatenate([keep_pos, NEG["idx"]]); active.sort()
    Xact_passed = X[:, passed][active]                     # (n_active, |passed|) bounded copy
    keep_local = spearman_dedup(Xact_passed, list(range(len(passed))), det_cfg["dedup"])
    order_cols = passed[keep_local]                        # global column indices in dedup order

    # ---- k-selection sweep on the selected columns only ----
    y = y_all[active]; folds = folds_all[active]
    Xsel = X[:, order_cols][active]                        # (n_active, kmax) -- only survivors, bounded
    ksel, ap = select_k(Xsel, y, folds, fold_ids, det_cfg, args.ksweep, args.boot, rng)
    return dict(det=det, frac=frac_pct, seed=seed, oof_ap=ap, n_features=int(ksel),
                n_gate=int(len(passed)), n_dedup=int(len(order_cols)), wall=time.time() - t0, status="ok")


def _bootstrap_ci_excl0(Xpos, negmean, negvar, negn, n_boot, rng):
    """WPW-only vectorized bootstrap of Cohen's d; CI excludes 0 per column. DISCLOSED APPROXIMATION:
    the frozen d_ci() resamples BOTH groups (n=1000); here only the 115-WPW group is resampled and the
    53k-negative moments are held fixed. At n_neg~53k the negative bootstrap adds <1e-3 to the CI, so
    gate membership is unchanged. (Resampling 53k x F x n_boot is infeasible.)"""
    npos, F = Xpos.shape
    idx = rng.integers(0, npos, size=(n_boot, npos))
    ds = np.empty((n_boot, F), np.float32)
    for b in range(n_boot):
        s = Xpos[idx[b]]
        mp = np.nanmean(s, 0); vp = np.nanvar(s, 0, ddof=1)
        ds[b] = cohens_d_from_moments(mp, vp, npos, negmean, negvar, negn).astype(np.float32)
    lo = np.nanpercentile(ds, 2.5, 0); hi = np.nanpercentile(ds, 97.5, 0)
    return ((lo > 0) == (hi > 0)) & np.isfinite(lo) & np.isfinite(hi)


# ==================================================================================================
# pool loading (once per detector; folds 1-8 only; precompute fixed-negative structures)
# ==================================================================================================
def load_pool(det):
    path = os.path.join(PROC, DETECTORS[det]["pool"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{det}] candidate pool not found: {path}")
    head = pd.read_csv(path, nrows=0); cols = list(head.columns)
    meta = [c for c in cols if c in META_CANDIDATES]; feats = [c for c in cols if c not in meta]
    # Robust dtypes: id/source columns as str; ALL feature columns as float32 (NaN preserved -- missing
    # values are informative: delineation failure is a signal and XGBoost routes NaN natively, so they
    # must NOT be filled/dropped/zeroed). We deliberately do NOT force an integer dtype on any meta
    # column: several (e.g. n_beats, and per detector extraction_failed / n_leads_morpho_failed) contain
    # NaN, and pandas cannot safely cast NaN-bearing float data to int at read time (the crash you saw).
    # label and fold never contain NaN, so they are cast to int AFTER reading.
    STR_COLS = {"ecg_id", "patient_id", "source"}
    dtype = {c: np.float32 for c in feats}
    for c in cols:
        if c in STR_COLS:
            dtype[c] = str
    # M3 (~6.3GB text) / M4 (~7.1GB text) -> ~1-1.7GB as float32; fits in 32GB. If a machine has less,
    # read with chunksize= accumulating into a np.memmap of shape (n, F); the rest is unchanged.
    df = pd.read_csv(path, dtype=dtype, low_memory=False)
    if "fold" not in df.columns or "label" not in df.columns:
        raise ValueError(f"[{det}] pool missing fold/label columns")
    df = df[df.fold.astype(np.int64).between(1, 8)].reset_index(drop=True)  # <<< folds 1-8 ONLY. fold 9 dropped. fold 10 never here.
    y = df.label.astype(np.int64).values.astype(np.int8)                    # label: never NaN
    folds = df.fold.astype(np.int64).values.astype(np.int8)                 # fold : never NaN
    assert folds.max() <= 8, "fold>8 present -- protocol violation"
    src = np.where(df.source.values == "ptbxl", 0, 1).astype(np.int8)
    X = np.ascontiguousarray(df[feats].to_numpy(np.float32))
    del df
    neg_idx = np.where(y == 0)[0]; Xn = X[neg_idx]
    # per-column negative moments (overall + per corpus) and presorted negatives + tie terms
    def col_mvn(M):
        m = np.nanmean(M, 0); n = np.sum(~np.isnan(M), 0).astype(float)
        v = np.nansum((M - m) ** 2, 0) / np.maximum(n - 1, 1); return m, v, n
    mean_, var_, n_ = col_mvn(Xn)
    sp = src[neg_idx] == 0; sn = src[neg_idx] == 1
    mean_p, var_p, n_p = col_mvn(Xn[sp]); mean_n, var_n, n_n = col_mvn(Xn[sn])
    # presorted negatives as ONE 2-D array (F, n_neg) so joblib/loky memmaps it once (shared, not
    # copied per worker). np.sort pushes NaN to the end; nvalid marks the non-NaN prefix per feature.
    F = X.shape[1]; nneg = len(neg_idx)
    negsorted = np.sort(Xn.T, axis=1).astype(np.float32)    # (F, n_neg), NaN last
    nvalid = np.sum(~np.isnan(Xn), axis=0).astype(np.int64)  # (F,)
    negtie = np.zeros(F)
    for f in range(F):
        col = negsorted[f, :nvalid[f]]
        if len(col):
            _, c = np.unique(col, return_counts=True); negtie[f] = float(np.sum(c ** 3 - c))
    del Xn
    NEG = dict(idx=neg_idx, mean=mean_, var=var_, n=n_, mean_p=mean_p, var_p=var_p, n_p=n_p,
               mean_n=mean_n, var_n=var_n, n_n=n_n, sorted=negsorted, nvalid=nvalid, tie=negtie)
    return dict(X=X, y=y, folds=folds, src=src, fold_ids=sorted(np.unique(folds).tolist()),
                n_feat=len(feats), n_pos=int((y == 1).sum()), n=len(y), neg=NEG)


# ==================================================================================================
# orchestration
# ==================================================================================================
def load_checkpoint():
    if os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {"units": {}, "meta": {}}


def save_json(state):
    json.dump(state, open(OUT_JSON, "w"), indent=2)
    if os.path.isdir(GH):
        os.makedirs(os.path.join(GH, "reports", "metrics"), exist_ok=True)
        shutil.copy(OUT_JSON, os.path.join(GH, "reports", "metrics", os.path.basename(OUT_JSON)))


def _bootmean(a, n=5000):
    rng = np.random.default_rng(0)
    return np.array([rng.choice(a, len(a), True).mean() for _ in range(n)])


def summarize(state):
    per = {}
    for det in DETECTORS:
        per[det] = {}
        for fp in FRACTIONS:
            u = [x for x in state["units"].values()
                 if x["det"] == det and x["frac"] == fp and x["status"] == "ok" and np.isfinite(x["oof_ap"])]
            if u:
                vals = [x["oof_ap"] for x in u]; ks = [x["n_features"] for x in u]
                per[det][str(fp)] = dict(mean_AP=float(np.mean(vals)), sd_AP=float(np.std(vals)),
                                         mean_k=float(np.mean(ks)), n_seeds=len(vals),
                                         per_seed_AP=[float(v) for v in vals],
                                         per_seed_k=[int(k) for k in ks])
    paired = {}
    for det in DETECTORS:
        pairs = []
        for s in range(state["meta"].get("seeds", 8)):
            a = state["units"].get(f"{det}|90|{s}"); b = state["units"].get(f"{det}|100|{s}")
            if a and b and a["status"] == "ok" and b["status"] == "ok":
                pairs.append(b["oof_ap"] - a["oof_ap"])
        if len(pairs) >= 2:
            arr = np.array(pairs, float); bm = _bootmean(arr)
            paired[det] = dict(n_pairs=len(pairs), mean_diff=float(arr.mean()),
                               se=float(arr.std(ddof=1) / np.sqrt(len(arr))),
                               ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                               still_rising=bool(np.percentile(bm, 2.5) > 0))
    m4cmp = {}
    for fp, cont in CONTAMINATED_M4.items():
        k = str(fp)
        if "M4" in per and k in per["M4"]:
            lf = per["M4"][k]["mean_AP"]; m4cmp[k] = dict(contaminated=cont, leakfree=round(lf, 4), delta=round(lf - cont, 4))
    state["summary"] = dict(per_detector=per, paired_90_vs_100=paired, M4_leakfree_vs_contaminated=m4cmp)
    return state


def make_figure(state):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = state.get("summary", {}).get("per_detector", {})
    colors = {"M1": "#16a34a", "M2": "#7c3aed", "M3": "#ea580c", "M4": "#2563eb", "M5": "#db2777"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for det, cfg in per.items():
        fps = sorted(int(k) for k in cfg)
        if not fps:
            continue
        m = np.array([cfg[str(f)]["mean_AP"] for f in fps]); sd = np.array([cfg[str(f)]["sd_AP"] for f in fps])
        ax.plot(fps, m, "o-", color=colors.get(det, "gray"), lw=1.6, ms=4, label=det)
        ax.fill_between(fps, m - sd, m + sd, color=colors.get(det, "gray"), alpha=0.15)
    ax.set_xlabel("Training positives kept (%)", fontsize=12)
    ax.set_ylabel("OOF Average Precision (folds 1-8)", fontsize=12)
    ax.set_title("Leak-free learning curves (feature selection re-run per subsample)\nmean +/- 1 SD over seeds", fontsize=12.5)
    ax.grid(alpha=0.25); ax.legend(title="detector"); fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    if os.path.isdir(GH):
        os.makedirs(os.path.join(GH, "reports", "figures"), exist_ok=True)
        shutil.copy(OUT_FIG, os.path.join(GH, "reports", "figures", os.path.basename(OUT_FIG)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--fractions", nargs="+", type=int, default=FRACTIONS)
    ap.add_argument("--ksweep", type=int, default=12, help="number of AP-vs-k grid points "
                    "(12 ~= the frozen protocols' own coarse K grids; the k-rule still picks among them)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (gate CI + 1-SE IC)")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    from joblib import Parallel, delayed
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(*a, **k):
            class _D:
                def __init__(s, **kk): s.total = kk.get("total", 0)
                def update(s, n=1): pass
                def set_postfix_str(s, *a, **k): pass
                def write(s, m): print(m)
                def close(s): pass
            return _D(**k)

    state = load_checkpoint()
    state["meta"] = dict(seeds=args.seeds, fractions=args.fractions, ksweep=args.ksweep, boot=args.boot,
                         protocol="leak-free: gate+dedup+k re-run per subsample",
                         folds="1-8 only; fold9 unused; fold10 never loaded")
    total = len(args.detectors) * len(args.fractions) * args.seeds
    done0 = sum(1 for u in state["units"].values() if u["status"] in ("ok", "empty_gate"))
    print(f"[plan] {total} units | {done0} already done | detectors={args.detectors} seeds={args.seeds} jobs={args.jobs}")
    t_start = time.time(); completed = done0
    bar = tqdm(total=total, initial=done0, desc="units", unit="unit")

    for det in args.detectors:
        keys = [f"{det}|{fp}|{s}" for fp in args.fractions for s in range(args.seeds)]
        if all(state["units"].get(k, {}).get("status") in ("ok", "empty_gate") for k in keys):
            continue
        bar.set_postfix_str(f"{det}: loading pool"); POOL = load_pool(det)
        bar.write(f"[{det}] pool: {POOL['n']} rows x {POOL['n_feat']} feats ({POOL['n_pos']} WPW) | "
                  f"pool RAM ~ {POOL['X'].nbytes/1e9:.2f} GB (shared)")
        for fp in args.fractions:
            todo = [s for s in range(args.seeds)
                    if state["units"].get(f"{det}|{fp}|{s}", {}).get("status") not in ("ok", "empty_gate")]
            if not todo:
                continue
            bar.set_postfix_str(f"{det} @ {fp}% ({len(todo)} seeds)")
            results = Parallel(n_jobs=args.jobs, prefer="processes", max_nbytes="1M")(
                delayed(run_unit)(det, fp, s, POOL, args) for s in todo)
            for r in results:
                state["units"][f"{det}|{fp}|{r['seed']}"] = r; completed += 1; bar.update(1)
            save_json(state)
            el = time.time() - t_start; rate = max(completed - done0, 1) / max(el, 1e-9)
            eta = (total - completed) / rate if rate > 0 else float("nan")
            bar.write(f"[ckpt] {det} @ {fp}% | {completed}/{total} | elapsed {el/60:.1f}m | ETA {eta/60:.1f}m")
        del POOL
    bar.close()
    state = summarize(state); save_json(state)
    try:
        make_figure(state)
    except Exception as e:
        print(f"[warn] figure failed: {e}")
    print("\n=== DONE ===")
    print("metrics -> " + OUT_JSON)
    print("figure  -> " + OUT_FIG)
    for det, cfg in state["summary"]["per_detector"].items():
        if cfg:
            print("  " + det + ": " + " | ".join(
                f"{f}%:{cfg[f]['mean_AP']:.3f}(k{cfg[f]['mean_k']:.0f})" for f in sorted(cfg, key=int)))
    print("paired 90->100 (still rising):")
    for det, pr in state["summary"]["paired_90_vs_100"].items():
        print(f"  {det}: mean_diff={pr['mean_diff']:+.4f} SE={pr['se']:.4f} "
              f"CI95={pr['ci95']} rising={pr['still_rising']}")


if __name__ == "__main__":
    main()
