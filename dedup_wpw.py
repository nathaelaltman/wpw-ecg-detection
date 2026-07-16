"""
dedup_wpw.py -- Q1: near-duplicate audit of the 142 WPW positives.

Pairwise similarity on the 0.5-40 Hz filtered signal, TWO complementary measures:
  (A) max normalized cross-correlation of the full 10 s lead-II trace (lag-tolerant),
  (B) cosine similarity of the *median beat* (P-QRS-T template), and separately
  (C) cosine on the -80 -> 0 ms window before R (the pre-excitation / delta zone).
A pair is flagged a near-duplicate if BOTH  max-xcorr >= THR  AND  median-beat cosine >= THR
(THR = 0.95). The delta-window cosine is reported alongside for every flagged pair (it is
the physiologically decisive window for WPW, so we show it but do not gate on it alone --
a genuinely duplicated record must match on the whole beat, not only the delta).

Outputs (printed + JSON + CSV):
  - every flagged pair with both scores, the delta-window cosine, both folds, both corpora,
    and age/sex of each member (exact-demographic matches are the strongest duplicate signal);
  - explicit fold-10 <-> folds 1-9 twin test: does any of the 14 fold-10 WPW have a
    near-twin outside fold 10 (would contaminate the sacred held-out set);
  - CSN split: of the 72 CSN WPW, how many come from the chapman-shaoxing sub-corpus vs
    ningbo, decided by the on-disk file prefix / rel_path.

Read-only: no model, no refit, fold assignments untouched. Loads only 142 signals (~seconds).

Run:  python dedup_wpw.py
Writes: reports/metrics/wpw_dedup.json
        reports/metrics/wpw_dedup_pairs.csv
"""
import os, sys, json, ast
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
sys.path.insert(0, SRC)
from signal_loading import load_signal, LEADS_CANONICAL, resolve_path

THR      = 0.95          # near-duplicate threshold for BOTH xcorr and median-beat cosine
FS       = 500
iII      = list(LEADS_CANONICAL).index("II")
PRE_MS, POST_MS = 250, 450     # median-beat window around R (matches fig05 beat extraction)
DELTA_MS = 80                  # -80 -> 0 ms pre-R window (pre-excitation zone)

# ---- 0.5-40 Hz filter (frozen), verbatim from the project ----
with open(os.path.join(PROC, "filter_config.json")) as f:
    FCFG = json.load(f)["filter_FINAL"]
SOS = butter(FCFG["order"], [FCFG["low"]/(FS/2), FCFG["high"]/(FS/2)], btype="band", output="sos")
def bpf(x): return sosfiltfilt(SOS, np.asarray(x, float))

# ---- R detection (Pan-Tompkins, verbatim from make_fig_qrs_tp_vs_fn.py) ----
def detect_r(sig):
    d = np.diff(sig, prepend=sig[0]); w = max(1, int(0.08*FS))
    mwi = np.convolve(d*d, np.ones(w)/w, mode="same")
    pk, _ = find_peaks(mwi, distance=int(0.30*FS), height=np.mean(mwi)+0.5*np.std(mwi)); R = []
    for p in pk:
        a = max(0, p-int(0.05*FS)); b = min(len(sig), p+int(0.05*FS))
        R.append(a+int(np.argmax(np.abs(sig[a:b]))))
    return np.array(sorted(set(R)))

def median_beat(sig, R):
    """Median P-QRS-T template on lead II, aligned on R. Returns (beat, delta_window)."""
    pre, post = int(PRE_MS/1000*FS), int(POST_MS/1000*FS)
    segs = []
    for r in R:
        a, b = r-pre, r+post
        if a < 0 or b > len(sig): continue
        segs.append(sig[a:b])
    if len(segs) < 1: return None, None
    beat = np.median(np.stack(segs), axis=0)
    dwin = int(DELTA_MS/1000*FS)
    delta = beat[pre-dwin:pre]            # -80 -> 0 ms before R
    return beat, delta

def znorm(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean())/s if s > 1e-9 else x - x.mean()

def max_xcorr(a, b):
    """Max normalized cross-correlation over lags (both z-normed)."""
    a, b = znorm(a), znorm(b)
    n = len(a)
    fc = np.correlate(a, b, mode="full") / n     # normalized by length (both unit-var)
    return float(np.max(fc))

def cosine(a, b):
    a, b = znorm(a), znorm(b)
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/d) if d > 1e-9 else np.nan

# ---- corpus of the 142 WPW positives ----
# Prefer the canonical combined metadata; fall back to any OOF file for label/fold/source.
def load_positives():
    cand = os.path.join(PROC, "metadata_combined.csv")
    if os.path.exists(cand):
        m = pd.read_csv(cand, dtype={"ecg_id": str})
    else:  # fall back to an OOF file (folds 1-8) UNION fold9/10 from per-corpus metadata
        m = pd.read_csv(os.path.join(PROC, "m3_combined_oof.csv"), dtype={"ecg_id": str})
    need = {"ecg_id", "source", "label", "fold"}
    assert need.issubset(m.columns), f"missing {need - set(m.columns)} in {cand}"
    pos = m[m.label == 1][["ecg_id", "source", "fold"]].drop_duplicates().reset_index(drop=True)
    return pos

# ---- age/sex per record (secondary signal for exact-demographic matches) ----
def load_demographics():
    dem = {}
    # PTB-XL
    ptb_db = None
    for root, _, files in os.walk(os.path.join(ROOT, "data", "raw", "ptbxl")):
        for fn in files:
            if fn == "ptbxl_database.csv": ptb_db = os.path.join(root, fn)
    if ptb_db:
        p = pd.read_csv(ptb_db)[["ecg_id", "age", "sex"]]
        p["ecg_id"] = p["ecg_id"].astype(int).astype(str)
        for _, r in p.iterrows():
            dem[("ptbxl", r.ecg_id)] = (r.age, "F" if r.sex == 1 else "M")
    # Ningbo/CSN
    nin = os.path.join(PROC, "metadata_ningbo.csv")
    if os.path.exists(nin):
        n = pd.read_csv(nin, dtype={"ecg_id": str})
        sc = "sex" if "sex" in n.columns else None
        ac = "age" if "age" in n.columns else None
        for _, r in n.iterrows():
            sx = str(r[sc])[:1].upper() if sc else "?"
            dem[("ningbo", str(r.ecg_id))] = (r[ac] if ac else np.nan, sx)
    return dem

def corpus_key(source):
    return "ptbxl" if "ptb" in str(source).lower() else "ningbo"

# ---- CSN sub-corpus (chapman-shaoxing vs ningbo) from the on-disk path ----
def csn_subcorpus(ecg_id, source):
    """Return 'chapman-shaoxing' | 'ningbo' | 'unknown' from the resolved path / file prefix."""
    if corpus_key(source) != "ningbo": return None
    try:
        p = resolve_path(ecg_id, source).lower().replace("\\", "/")
    except Exception:
        p = str(ecg_id).lower()
    base = os.path.basename(p)
    # Chapman-Shaoxing records are the JS##### series; the physionet ningbo release also uses
    # a directory partition. Decide first on any explicit directory tag, then on the file prefix.
    if "chapman" in p or "shaoxing" in p: return "chapman-shaoxing"
    if "ningbo" in p.replace("a-large-scale", ""):  # avoid matching the dataset root folder name
        # only count 'ningbo' if it appears in a sub-path segment, not the release root
        segs = [s for s in p.split("/") if s and "a-large-scale-12-lead" not in s]
        if any("ningbo" in s for s in segs): return "ningbo"
    # file-prefix heuristic: JS = Chapman-Shaoxing convention; g#/HR#### seen in Ningbo-native
    if base.startswith("js"): return "chapman-shaoxing"
    if base and base[0] == "g": return "ningbo"
    return "unknown"

def main():
    pos = load_positives()
    assert len(pos) == 142, f"expected 142 WPW, got {len(pos)}"
    dem = load_demographics()

    # load + filter + median-beat for each positive
    recs = []
    for _, r in pos.iterrows():
        try:
            sig = bpf(load_signal(r.ecg_id, r.source)[:, iII])
        except Exception as e:
            print(f"  WARN load failed {r.source}/{r.ecg_id}: {e}"); continue
        R = detect_r(sig)
        beat, delta = median_beat(sig, R)
        ck = corpus_key(r.source)
        age, sex = dem.get((ck, str(r.ecg_id)), (np.nan, "?"))
        recs.append(dict(ecg_id=str(r.ecg_id), source=r.source, corpus=ck, fold=int(r.fold),
                         age=age, sex=sex, sub=csn_subcorpus(r.ecg_id, r.source),
                         sig=sig, beat=beat, delta=delta))
    n = len(recs)
    print(f"Loaded {n}/142 WPW signals (filtered, lead II).")

    # pairwise
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            a, b = recs[i], recs[j]
            xc = max_xcorr(a["sig"], b["sig"])
            cb = cosine(a["beat"], b["beat"]) if (a["beat"] is not None and b["beat"] is not None) else np.nan
            cd = cosine(a["delta"], b["delta"]) if (a["delta"] is not None and b["delta"] is not None) else np.nan
            if np.isfinite(xc) and np.isfinite(cb) and xc >= THR and cb >= THR:
                pairs.append(dict(
                    id_a=a["ecg_id"], src_a=a["source"], fold_a=a["fold"], age_a=a["age"], sex_a=a["sex"],
                    id_b=b["ecg_id"], src_b=b["source"], fold_b=b["fold"], age_b=b["age"], sex_b=b["sex"],
                    max_xcorr=round(xc, 4), medbeat_cos=round(cb, 4), delta_cos=round(float(cd), 4),
                    same_corpus=int(a["corpus"] == b["corpus"]),
                    demo_exact=int(a["age"] == b["age"] and a["sex"] == b["sex"] and str(a["age"]) != "nan")))
    pairs.sort(key=lambda d: -d["max_xcorr"])

    # fold-10 twin test
    f10 = [r for r in recs if r["fold"] == 10]
    f10_twins = []
    for p in pairs:
        fa, fb = p["fold_a"], p["fold_b"]
        if (fa == 10) ^ (fb == 10):   # exactly one member in fold 10
            outside = p["id_b"] if fa == 10 else p["id_a"]
            inside  = p["id_a"] if fa == 10 else p["id_b"]
            f10_twins.append(dict(fold10_id=inside, twin_id=outside, twin_fold=(fb if fa == 10 else fa),
                                  max_xcorr=p["max_xcorr"], medbeat_cos=p["medbeat_cos"]))

    # CSN sub-corpus split
    csn = [r for r in recs if r["corpus"] == "ningbo"]
    split = {}
    for r in csn:
        split[r["sub"]] = split.get(r["sub"], 0) + 1

    # report
    print("\n=== NEAR-DUPLICATE PAIRS (xcorr>=%.2f AND median-beat cos>=%.2f) ===" % (THR, THR))
    if not pairs:
        print("  none.")
    for p in pairs:
        tag = "  [EXACT age+sex]" if p["demo_exact"] else ""
        print("  %s/%s (fold%d, %s/%s) <-> %s/%s (fold%d, %s/%s) | xcorr=%.3f beat=%.3f delta=%.3f%s" % (
            p["src_a"], p["id_a"], p["fold_a"], str(p["age_a"]), p["sex_a"],
            p["src_b"], p["id_b"], p["fold_b"], str(p["age_b"]), p["sex_b"],
            p["max_xcorr"], p["medbeat_cos"], p["delta_cos"], tag))

    print("\n=== FOLD-10 TWIN TEST (contamination of the sacred held-out set) ===")
    print("  fold-10 WPW count: %d" % len(f10))
    if not f10_twins:
        print("  NO fold-10 WPW has a near-twin in folds 1-9. Held-out set clean on this criterion.")
    else:
        print("  WARNING: fold-10 records with a near-twin outside fold 10:")
        for t in f10_twins:
            print("    fold10 %s  <->  fold%d %s   (xcorr=%.3f beat=%.3f)" % (
                t["fold10_id"], t["twin_fold"], t["twin_id"], t["max_xcorr"], t["medbeat_cos"]))

    print("\n=== CSN (72 WPW) sub-corpus split ===")
    for k, v in sorted(split.items(), key=lambda x: -x[1]):
        print("  %-18s %d" % (str(k), v))
    print("  (total CSN WPW = %d)" % len(csn))

    out = dict(threshold=THR, n_loaded=n, n_pairs=len(pairs), pairs=pairs,
               fold10_wpw=len(f10), fold10_twins=f10_twins, csn_split=split, csn_total=len(csn))
    json.dump(out, open(os.path.join(MET, "wpw_dedup.json"), "w"), indent=2, default=str)
    pd.DataFrame(pairs).to_csv(os.path.join(MET, "wpw_dedup_pairs.csv"), index=False)
    print("\nWrote reports/metrics/wpw_dedup.json  and  reports/metrics/wpw_dedup_pairs.csv")

if __name__ == "__main__":
    main()
