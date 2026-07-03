"""
shap_explain.py -- per-ECG SHAP explanation for the frozen feature-based detectors (M1-M5).

Reusable helper for the paper and the Flask interface: given a frozen XGBoost detector
and one ECG's feature row, return the features pushing the WPW score up and down.
Exact TreeSHAP via shap.TreeExplainer, with a native-xgboost fallback (identical values,
computed by the booster) for cross-version robustness. No refit, read-only.

For the deep detector M7, use Grad-CAM instead (SHAP on a raw 12x5000 signal is not readable).

Main entry point:
    explain_ecg(model, x_row, feature_names=None, top_k=5) -> dict
"""
from pathlib import Path
import sys
import numpy as np


def tree_shap(model, X):
    """Exact TreeSHAP values (n, k) + scalar base value. shap.TreeExplainer first;
    on any loader/version error, fall back to xgboost's own pred_contribs (same math)."""
    import numpy as np
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Xv = Xv.astype("float32")
    try:
        import shap
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xv)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                       # (n, k, classes)
            sv = sv[..., -1]
        base = expl.expected_value
        base = float(np.asarray(base).reshape(-1)[-1]) if isinstance(base, (list, np.ndarray)) else float(base)
        return sv, base
    except Exception:
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        fnames = list(X.columns) if hasattr(X, "columns") else getattr(booster, "feature_names", None)
        if fnames is not None and len(fnames) != Xv.shape[1]:
            fnames = None
        dm = xgb.DMatrix(Xv, missing=np.nan, feature_names=fnames)
        contribs = booster.predict(dm, pred_contribs=True)   # (n, k+1), last col = bias
        return contribs[:, :-1], float(np.mean(contribs[:, -1]))


def explain_ecg(model, x_row, feature_names=None, top_k=5):
    """
    Explain one ECG's WPW score from a frozen tree detector.

    Parameters
    ----------
    model         : a fitted XGBoost model (raw/uncalibrated; explains the ranking score).
    x_row         : the ECG's feature row aligned to the model's selected features
                    (pandas Series with feature names, or a 1D array in feature order).
    feature_names : required if x_row is a plain array; taken from the Series index otherwise.
    top_k         : number of up- and down-pushing features to return.

    Returns
    -------
    dict:
      base_value   : the model's expected score (log-odds margin).
      pushing_up   : [{feature, shap, direction='up'}], strongest positive contributions.
      pushing_down : [{feature, shap, direction='down'}], strongest negative contributions.
      sum_shap     : total SHAP contribution (score margin minus base_value).
    """
    if hasattr(x_row, "index"):
        if feature_names is None:
            feature_names = list(x_row.index)
        x = x_row.values.astype("float32").reshape(1, -1)
    else:
        x = np.asarray(x_row, dtype="float32").reshape(1, -1)
    sv, base = tree_shap(model, x)
    sv = np.asarray(sv).reshape(-1)
    if feature_names is None:
        feature_names = ["f%d" % i for i in range(len(sv))]
    order = np.argsort(-np.abs(sv))
    up = [dict(feature=feature_names[i], shap=round(float(sv[i]), 5), direction="up")
          for i in order if sv[i] > 0][:top_k]
    down = [dict(feature=feature_names[i], shap=round(float(sv[i]), 5), direction="down")
            for i in order if sv[i] < 0][:top_k]
    return dict(base_value=round(base, 5), pushing_up=up, pushing_down=down,
                sum_shap=round(float(sv.sum()), 5))


if __name__ == "__main__":
    # Demo on one frozen detector + one saved example row (no big feature matrix needed).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root (config.py)
    import json, joblib
    from config import MODELS, METRICS
    try:
        model = joblib.load(MODELS / "M4_medianbeat" / "m4_combined_wavelet_env_model_raw.joblib")
        ex = json.load(open(METRICS / "shap_example_M4.json", encoding="utf-8"))
        out = explain_ecg(model, np.asarray(ex["values"], dtype="float32"),
                          feature_names=ex["features"], top_k=5)
        print("Demo explain_ecg (M4, one example ECG):")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("Demo skipped (%s). explain_ecg is importable and ready for the Flask output." % e)
