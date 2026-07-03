# -*- coding: utf-8 -*-
"""
src/evaluation.py — STANDARDIZED evaluation of the WPW models (single source of truth).

Every model (M1..M7) is built its own way but evaluated IDENTICALLY through
`evaluate_standard`, which always produces the same 3 outputs:
  A. a metrics dict (fixed keys)
  B. a figure {model}_evaluation.png (confusion matrix + PR + ROC + F1-vs-threshold)
  C. a file {model}_metrics.json

Fixed protocol:
  - threshold = F1-max computed on the OOF training folds (1-8), then APPLIED to the held-out fold
  - AP and AUC with bootstrap 95% CI; optional multi-seed stability
  - confusion + P/R/F1/specificity/NPV at the threshold, on the held-out fold

Note: the held-out fold here is the validation fold (fold 9). The sacred test fold (fold 10) is
never touched until the final ensemble evaluation.

Two DISTINCT gaps are recorded (do not conflate):
  - gap_train_fold9 = ap_train - AP(fold9)      (overfit vs the held-out fold shown here)
  - gap_train_oof   = ap_train - ap_oof         (the selection gap used to pick K; needs ap_oof)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve, confusion_matrix,
                             brier_score_loss)


# ──────────────────────────────────────────────────────────────────────────────
def _boot_ci(y_true, scores, metric_fn, n=2000, seed=42):
    """Stratified bootstrap 95% CI (resamples positives and negatives separately)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)
    out = np.empty(n)
    for i in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        out[i] = metric_fn(y_true[idx], scores[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def f1max_threshold(y_oof, score_oof):
    """F1-max threshold determined on the OOF training folds 1-8 (robust: many WPW there vs few in the held-out fold)."""
    prec, rec, thr = precision_recall_curve(y_oof, score_oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    bi = int(np.argmax(f1[:-1]))
    return float(thr[bi])


def confusion_metrics(y_true, pred):
    """TP/FP/FN/TN + precision/recall/F1/specificity/NPV from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    P  = tp / (tp + fp) if (tp + fp) else 0.0          # precision
    R  = tp / (tp + fn) if (tp + fn) else 0.0          # recall / sensitivity
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    SP = tn / (tn + fp) if (tn + fp) else 0.0          # specificity
    NPV = tn / (tn + fn) if (tn + fn) else 0.0
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision=float(P), recall=float(R), f1=float(F1),
                specificity=float(SP), npv=float(NPV))


# ──────────────────────────────────────────────────────────────────────────────
def write_oof_canonical(out_path, df=None, ecg_id=None, source=None, fold=None,
                        label=None, proba_raw=None, proba_cal=None):
    """
    Write an OOF file to the CANONICAL schema (single source of truth for the ensemble):
        columns = ['ecg_id', 'source', 'fold', 'label', 'proba_raw', 'proba_cal']  (this order)

    Input : either a (partial) DataFrame `df`, OR the separate column arrays.
    `proba_cal` defaults to NaN (e.g. M7 has no calibrated score) — the column ALWAYS exists.
    `ecg_id` is forced to str (alignment is always by (ecg_id, source), never row order).
    Idempotent (overwrites). Does NO score computation — formatting + write only.
    """
    cols = ['ecg_id', 'source', 'fold', 'label', 'proba_raw', 'proba_cal']
    if df is not None:
        d = pd.DataFrame(df).copy()
        if proba_cal is not None and 'proba_cal' not in d.columns:
            d['proba_cal'] = np.asarray(proba_cal)
    else:
        d = pd.DataFrame({'ecg_id': np.asarray(ecg_id), 'source': np.asarray(source),
                          'fold': np.asarray(fold), 'label': np.asarray(label),
                          'proba_raw': np.asarray(proba_raw)})
        d['proba_cal'] = np.nan if proba_cal is None else np.asarray(proba_cal)
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    d['ecg_id'] = d['ecg_id'].astype(str)
    d = d[cols]
    d.to_csv(out_path, index=False)
    return d


def permutation_control(X, y, folds, make_model_fn, n_shuffle=5,
                        metric_fn=average_precision_score, seed=123):
    """
    Standardized negative control (leak sentinel). For each label shuffle, rebuild a
    NATIVE-fold OOF and score it. Returns {'null_mean','null_max','n_shuffle'}.
    On real signal the null AP collapses toward prevalence. Reusable (retroactive leakage sentinel on M1-M4).

    `make_model_fn()` must return a FRESH unfitted estimator with fit/predict_proba.
    """
    X = np.asarray(X); y = np.asarray(y); folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    ufolds = np.unique(folds)
    nulls = []
    for _ in range(int(n_shuffle)):
        yperm = rng.permutation(y)
        oof = np.full(len(y), np.nan)
        for f in ufolds:
            te = folds == f; tr = ~te
            if yperm[tr].sum() < 1 or (yperm[tr] == 0).sum() < 1:
                continue
            m = make_model_fn(); m.fit(X[tr], yperm[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof)
        nulls.append(float(metric_fn(yperm[ok], oof[ok])))
    return {'null_mean': float(np.mean(nulls)), 'null_max': float(np.max(nulls)),
            'n_shuffle': int(n_shuffle)}


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_standard(name, y_oof, score_oof, y_test, score_test,
                      figures_dir, metrics_dir,
                      score_test_calibrated=None, ap_train=None, ap_oof=None,
                      multiseed=None, extra=None):
    """
    Standardized evaluation of a model.

    Parameters
    ----------
    name            : str, short id (e.g. 'M2_combined'). Used for output filenames.
    y_oof, score_oof: labels and OOF scores on training folds 1-8 (for the F1-max threshold).
    y_test, score_test : labels and RAW scores of the held-out fold (fold 9).
    figures_dir, metrics_dir : output folders.
    score_test_calibrated : calibrated scores of the held-out fold (for Brier), optional.
    ap_train        : resubstitution AP (for the gaps), optional.
    ap_oof          : OOF AP folds 1-8 (for `gap_train_oof`, the selection gap), optional.
    multiseed       : dict {'AP_mean','AP_std','AUC_mean','AUC_std'}, optional.
    extra           : dict of extra fields to store in the JSON, optional.

    Returns
    -------
    metrics : dict (output A). Also writes B (PNG) and C (JSON).
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    y_oof = np.asarray(y_oof); score_oof = np.asarray(score_oof)
    y_test = np.asarray(y_test); score_test = np.asarray(score_test)

    # ── A. metrics ────────────────────────────────────────────────────────────
    AP  = float(average_precision_score(y_test, score_test))
    AUC = float(roc_auc_score(y_test, score_test))
    ap_lo, ap_hi   = _boot_ci(y_test, score_test, average_precision_score)
    auc_lo, auc_hi = _boot_ci(y_test, score_test, roc_auc_score)

    THR  = f1max_threshold(y_oof, score_oof)          # threshold on OOF, applied to the held-out fold
    pred = (score_test >= THR).astype(int)
    cm   = confusion_metrics(y_test, pred)

    brier = (float(brier_score_loss(y_test, score_test_calibrated))
             if score_test_calibrated is not None else None)

    metrics = {
        "model": name,
        "n_test": int(len(y_test)), "n_test_pos": int(y_test.sum()),
        "AP": AP, "AP_IC95": [ap_lo, ap_hi],
        "AUC": AUC, "AUC_IC95": [auc_lo, auc_hi],
        "threshold_F1max_on_OOF": THR,
        "confusion_at_threshold": cm,
        "brier_calibrated": brier,
        "ap_train": (float(ap_train) if ap_train is not None else None),
        "ap_oof": (float(ap_oof) if ap_oof is not None else None),
        # Two DISTINCT gaps — do not conflate:
        "gap_train_fold9": (float(ap_train - AP) if ap_train is not None else None),
        "gap_train_oof": (float(ap_train - ap_oof)
                          if (ap_train is not None and ap_oof is not None) else None),
        "multiseed": multiseed,
    }
    if extra:
        metrics.update(extra)

    # ── B. standardized figure ────────────────────────────────────────────────
    prec, rec, thr_pr = precision_recall_curve(y_test, score_test)
    f1c = 2 * prec * rec / (prec + rec + 1e-12)
    fpr, tpr, _ = roc_curve(y_test, score_test)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (1) confusion matrix
    M = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
    im = ax[0, 0].imshow(M, cmap='Blues')
    for (i, j), v in np.ndenumerate(M):
        ax[0, 0].text(j, i, f'{v}', ha='center', va='center', fontsize=18,
                      color='white' if v > M.max() / 2 else 'black')
    ax[0, 0].set_xticks([0, 1]); ax[0, 0].set_xticklabels(['Pred. non-WPW', 'Pred. WPW'])
    ax[0, 0].set_yticks([0, 1]); ax[0, 0].set_yticklabels(['True non-WPW', 'True WPW'])
    ax[0, 0].set_title(f"Confusion (F1-max threshold={THR:.3f})\n"
                       f"P={cm['precision']:.2f} R={cm['recall']:.2f} F1={cm['f1']:.2f} "
                       f"Spec={cm['specificity']:.3f}", fontsize=10)

    # (2) PR with operating point
    ax[0, 1].plot(rec, prec, color='#2563eb')
    bi = int(np.argmax(f1c[:-1]))
    ax[0, 1].scatter(rec[bi], prec[bi], c='r', zorder=5)
    ax[0, 1].set(title=f'PR (AP={AP:.3f} CI95[{ap_lo:.2f},{ap_hi:.2f}])',
                 xlabel='Recall', ylabel='Precision'); ax[0, 1].grid(alpha=.3)

    # (3) ROC
    ax[1, 0].plot(fpr, tpr, color='#2563eb'); ax[1, 0].plot([0, 1], [0, 1], '--', color='gray')
    ax[1, 0].set(title=f'ROC (AUC={AUC:.3f} CI95[{auc_lo:.2f},{auc_hi:.2f}])',
                 xlabel='FPR', ylabel='TPR'); ax[1, 0].grid(alpha=.3)

    # (4) F1 vs threshold
    ax[1, 1].plot(thr_pr, f1c[:-1], color='#16a34a')
    ax[1, 1].axvline(THR, ls='--', color='r', label=f'F1-max threshold (OOF)={THR:.3f}')
    ax[1, 1].set(title='F1 vs threshold (held-out fold)', xlabel='threshold', ylabel='F1')
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)

    ttl = f'{name} — standardized evaluation (held-out fold, {int(y_test.sum())} WPW)'
    if multiseed:
        ttl += f"   |   AP={multiseed['AP_mean']:.3f}±{multiseed['AP_std']:.3f}"
    plt.suptitle(ttl); plt.tight_layout()
    fig_path = os.path.join(figures_dir, f'{name}_evaluation.png')
    plt.savefig(fig_path, dpi=140, bbox_inches='tight'); plt.show()

    # ── C. metrics file ───────────────────────────────────────────────────────
    json_path = os.path.join(metrics_dir, f'{name}_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # console summary
    print(f"=== {name} — standardized evaluation ===")
    print(f"  AP  {AP:.3f} CI95[{ap_lo:.3f},{ap_hi:.3f}]"
          + (f" | multiseed {multiseed['AP_mean']:.3f}±{multiseed['AP_std']:.3f}" if multiseed else ""))
    print(f"  AUC {AUC:.3f} CI95[{auc_lo:.3f},{auc_hi:.3f}]")
    print(f"  F1-max threshold (OOF) = {THR:.4f}")
    print(f"  held-out fold @threshold : TP {cm['TP']} FP {cm['FP']} FN {cm['FN']} TN {cm['TN']}")
    print(f"    precision {cm['precision']:.3f} | recall {cm['recall']:.3f} | F1 {cm['f1']:.3f} "
          f"| specificity {cm['specificity']:.3f} | NPV {cm['npv']:.3f}")
    if brier is not None:
        print(f"  Brier (calibrated) {brier:.5f}")
    if ap_train is not None:
        print(f"  gap train-fold9 {ap_train - AP:+.3f}"
              + (f" | gap train-OOF {ap_train - ap_oof:+.3f}" if ap_oof is not None else ""))
    print(f"  -> figure: {fig_path}")
    print(f"  -> metrics: {json_path}")
    return metrics
