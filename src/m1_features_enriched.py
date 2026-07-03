"""
m1_features_enriched.py — ENRICHED M1 feature extractor (DOCUMENTATION / rejected hypothesis).

This is the extractor for the documented enrichment experiment
(`notebooks/05_m1_enrichment_rejected.ipynb`). It extends the canonical v1 pool
(`m1_features.py`) with, per lead II/V1/V5:
  - per-beat DISTRIBUTION summaries of every morphology base (med / std / IQR / skew) instead of the
    median only  -> captures beat-to-beat variability, which is ORTHOGONAL to M2 (global, no beats)
    and M4 (a single median beat);
  - extra morphology bases (delta concavity / time-to-half / smoothness, QRS width at 25/50/75 %,
    R-sharpness, Higuchi fractal dim, up/down slopes, ST slopes at J+40/J+80, T up/down slopes,
    T peak-to-end);
  - QRS-delta window SUB-SEGMENTS (slope + area over thirds of the onset window).

Verdict (see the DOC notebook): the enrichment does NOT lift M1's ceiling. Under a disciplined
selection (depth-2, gap<=0.30) the enriched pool ties the v1 pool on OOF AP and is slightly worse on
AUC; gains appear only at depth-3 / large-gap (memorization). M1 is signal-limited: the WPW delta wave
destabilizes NeuroKit delineation, so adding features cannot raise the ceiling. The CANONICAL M1 uses
the simpler v1 pool (`m1_features.py`).

NOTE ON REPRODUCIBILITY: the frozen enriched table used by the DOC notebook is the archived
`data/processed/m1_features_enriched.csv` (gitignored, ~700 features). The DOC notebook loads it via the
build guard and never re-runs extraction. This module is a faithful re-implementation of that extractor
(the original was not under version control); regenerating from scratch reproduces the same feature
*families* and rigor, which is what the experiment documents.

Main entry point:
    build_m1_features(meta, out_csv, force=False) -> pandas.DataFrame   (same signature as m1_features.py)
"""
import os, json, warnings, time, contextlib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis
from signal_loading import load_signal, LEADS_CANONICAL

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root (contains config.py)
from config import PROCESSED

with open(os.path.join(PROCESSED, "filter_config.json"), encoding="utf-8") as _f:
    _FCFG = json.load(_f)["filter_FINAL"]
FS = _FCFG["fs"]
_SOS = butter(_FCFG["order"], [_FCFG["low"]/(FS/2), _FCFG["high"]/(FS/2)], btype="band", output="sos")
def _bp(x):
    return sosfiltfilt(_SOS, np.asarray(x, dtype=np.float64))

LEADS_M1 = ["II", "V1", "V5"]
LEAD_IDX = {L: LEADS_CANONICAL.index(L) for L in LEADS_M1}

# Per-beat distribution summaries applied to every morphology base (this is the enrichment vs v1).
_SUMM = ("med", "std", "iqr", "skew")
def _summarize(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {s: np.nan for s in _SUMM}
    return {
        "med":  float(np.median(v)),
        "std":  float(np.std(v)) if v.size >= 2 else np.nan,
        "iqr":  float(np.percentile(v, 75) - np.percentile(v, 25)) if v.size >= 2 else np.nan,
        "skew": float(skew(v)) if v.size >= 3 else np.nan,
    }
def _med(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.median(v)) if v.size else np.nan
def _std(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.std(v)) if v.size >= 2 else np.nan


def _delineate(s):
    import neurokit2 as nk
    _, info = nk.ecg_peaks(s, sampling_rate=FS); rp = info["ECG_R_Peaks"]
    if rp is None or len(rp) < 3:
        return None, None
    _, w = nk.ecg_delineate(s, rpeaks=rp, sampling_rate=FS, method="dwt")
    return np.asarray(rp), w


# ── Clinical pool (identical to the canonical v1 extractor) ────────────────────────────────────────
CLIN = ['PR_ms','QRS_ms','QT_ms','P_ms','PR_seg_ms','ST_seg_ms','RR_mean','RR_std','HR','n_beats',
        'QRS_ms_std','QT_ms_std','P_amp_glob']

def _clinical(s):
    o = {k: np.nan for k in CLIN}
    rp, w = _delineate(s)
    if rp is None:
        return o, True
    rpf = np.asarray(rp, float); rr = np.diff(rpf)/FS*1000
    o['RR_mean'] = float(np.nanmean(rr)) if rr.size else np.nan
    o['RR_std']  = float(np.nanstd(rr)) if rr.size else np.nan
    o['HR']      = 60000/o['RR_mean'] if o['RR_mean'] and o['RR_mean'] > 0 else np.nan
    o['n_beats'] = int(len(rpf))
    Pon = np.asarray(w.get('ECG_P_Onsets', []), float); Poff = np.asarray(w.get('ECG_P_Offsets', []), float)
    Ppk = np.asarray(w.get('ECG_P_Peaks', []), float)
    Ron = np.asarray(w.get('ECG_R_Onsets', []), float); Roff = np.asarray(w.get('ECG_R_Offsets', []), float)
    Toff = np.asarray(w.get('ECG_T_Offsets', []), float); Tpk = np.asarray(w.get('ECG_T_Peaks', []), float)
    n = min(len(Pon), len(Ron))
    if n: o['PR_ms'] = _med([(Ron[i]-Pon[i])/FS*1000 for i in range(n) if np.isfinite(Pon[i]) and np.isfinite(Ron[i]) and Ron[i] > Pon[i]])
    m = min(len(Ron), len(Roff))
    if m:
        qrs = [(Roff[i]-Ron[i])/FS*1000 for i in range(m) if np.isfinite(Ron[i]) and np.isfinite(Roff[i]) and Roff[i] > Ron[i]]
        o['QRS_ms'] = _med(qrs); o['QRS_ms_std'] = _std(qrs)
    k = min(len(Ron), len(Toff))
    if k:
        qt = [(Toff[i]-Ron[i])/FS*1000 for i in range(k) if np.isfinite(Ron[i]) and np.isfinite(Toff[i]) and Toff[i] > Ron[i]]
        o['QT_ms'] = _med(qt); o['QT_ms_std'] = _std(qt)
    p = min(len(Pon), len(Poff))
    if p: o['P_ms'] = _med([(Poff[i]-Pon[i])/FS*1000 for i in range(p) if np.isfinite(Pon[i]) and np.isfinite(Poff[i]) and Poff[i] > Pon[i]])
    ps = min(len(Poff), len(Ron))
    if ps: o['PR_seg_ms'] = _med([(Ron[i]-Poff[i])/FS*1000 for i in range(ps) if np.isfinite(Poff[i]) and np.isfinite(Ron[i]) and Ron[i] > Poff[i]])
    st = min(len(Roff), len(Tpk))
    if st: o['ST_seg_ms'] = _med([(Tpk[i]-Roff[i])/FS*1000 for i in range(st) if np.isfinite(Roff[i]) and np.isfinite(Tpk[i]) and Tpk[i] > Roff[i]])
    pk = Ppk[np.isfinite(Ppk)].astype(int); pk = pk[(pk >= 0) & (pk < len(s))]
    o['P_amp_glob'] = _med(s[pk]) if pk.size else np.nan
    return o, False


def _higuchi(x, kmax=6):
    """Higuchi fractal dimension of a short signal segment (texture of the QRS-delta window)."""
    x = np.asarray(x, float); N = len(x)
    if N < 2*kmax + 2:
        return np.nan
    L = []
    for k in range(1, kmax+1):
        Lk = []
        for msta in range(k):
            idx = np.arange(msta, N, k)
            if len(idx) < 2: continue
            lmk = np.sum(np.abs(np.diff(x[idx]))) * (N-1) / (((len(idx)-1)) * k)
            Lk.append(lmk)
        if Lk: L.append((np.log(k), np.log(np.mean(Lk))))
    if len(L) < 2:
        return np.nan
    L = np.array(L)
    return float(-np.polyfit(L[:,0], L[:,1], 1)[0])


def _beat_morpho(s, r, Ron_i, Roff_i, Tpk_i, Toff_i):
    """Per-beat morphology bases (v1 set + enrichment bases). Returns {base: value or nan}."""
    o = {}
    win40 = int(0.040*FS)
    a = max(0, r-win40)
    if r > a+3:
        seg = s[a:r+1]; d1 = np.diff(seg)*FS; absd = np.abs(d1)
        # delta-onset velocity family
        for ms, key in [(20,'20'),(40,'40'),(60,'60')]:
            npt = int(ms/1000*FS); segk = s[max(0, r-npt):r+1]
            o[f'delta_slopemax{key}'] = float(np.max(np.abs(np.diff(segk)*FS))) if segk.size > 2 else np.nan
        o['delta_slopemean'] = float(absd.mean())
        imax = int(np.argmax(absd)); o['delta_t_to_max'] = imax/FS*1000
        init = np.mean(absd[:max(1, len(absd)//3)]); pk = absd.max()
        o['delta_ratio_init_peak'] = float(init/pk) if pk > 1e-9 else np.nan
        d2 = np.diff(d1)*FS; o['delta_accel_max'] = float(np.max(np.abs(d2))) if len(d2) else np.nan
        ideal = np.linspace(seg[0], seg[-1], len(seg)); o['delta_empatement_area'] = float(np.trapezoid(np.abs(seg-ideal))/FS)
        thr = 0.5*pk; idx = np.where(absd >= thr)[0]; o['delta_slow_phase_ms'] = float(idx[0]/FS*1000) if idx.size else np.nan
        # enrichment: time-to-half-amplitude, concavity, smoothness
        amp = seg - seg[0]; half = amp[-1]/2 if abs(amp[-1]) > 1e-9 else np.nan
        if np.isfinite(half) and abs(amp[-1]) > 1e-9:
            cross = np.where(np.sign(amp - half) != np.sign(amp[0] - half))[0]
            o['delta_time_to_half'] = float(cross[0]/FS*1000) if cross.size else np.nan
        else:
            o['delta_time_to_half'] = np.nan
        o['delta_concavity'] = float(np.mean(d2)) if len(d2) else np.nan          # mean 2nd-derivative (bow)
        o['delta_smoothness'] = float(np.std(d2)) if len(d2) >= 2 else np.nan      # jerk
        # QRS shape family
        rng = seg.max()-seg.min()
        o['qrs_area'] = float(np.trapezoid(seg-seg[0])/FS); o['qrs_absarea'] = float(np.trapezoid(np.abs(seg-seg[0]))/FS)
        o['qrs_energy'] = float(np.sum(seg**2))
        dd2 = np.diff(np.sign(np.diff(seg))); o['qrs_ninflect'] = int(np.sum(dd2 != 0))
        o['qrs_nzero'] = int(np.sum(np.diff(np.sign(seg-seg.mean())) != 0))
        o['qrs_skew'] = float(skew(seg)) if seg.size > 3 else np.nan
        o['qrs_kurt'] = float(kurtosis(seg)) if seg.size > 3 else np.nan
        o['qrs_p2p'] = float(rng)
        hlev = seg.min()+rng/2; above = np.where(seg >= hlev)[0]
        o['qrs_fwhm'] = float((above[-1]-above[0])/FS*1000) if above.size > 1 else np.nan
        pkp = int(np.argmax(np.abs(seg-seg[0]))); o['qrs_asym'] = pkp/(len(seg)-pkp) if len(seg)-pkp > 0 else np.nan
        o['qrs_arclen'] = float(np.sum(np.sqrt(1+np.diff(seg)**2)))
        pos = seg[seg > seg.mean()].sum(); negv = abs(seg[seg < seg.mean()].sum())
        o['qrs_posneg_ratio'] = float(pos/negv) if negv > 1e-9 else np.nan
        ddp = np.diff(np.sign(np.diff(seg))); o['qrs_npeaks'] = int(np.sum(ddp < 0))
        o['qrs_moment3'] = float(np.mean((seg-seg.mean())**3))
        ppp = np.abs(seg-seg.min())+1e-9; ppp = ppp/ppp.sum(); o['qrs_entropy'] = float(-np.sum(ppp*np.log(ppp)))
        # enrichment: width at 25/50/75 % of peak, R-sharpness, Higuchi
        for frac, key in [(0.25,'25'),(0.50,'50'),(0.75,'75')]:
            lev = seg.min() + frac*rng; ab = np.where(seg >= lev)[0]
            o[f'qrs_width{key}'] = float((ab[-1]-ab[0])/FS*1000) if ab.size > 1 else np.nan
        o['qrs_Rsharp'] = float(np.max(np.abs(d2))) if len(d2) else np.nan        # curvature at the peak
        o['qrs_higuchi'] = _higuchi(seg)
        # enrichment: up/down slopes and arc-length of the rising limb
        ip = int(np.argmax(seg))
        o['up_slope']   = float((seg[ip]-seg[0])/((ip)/FS)) if ip > 0 else np.nan
        o['down_slope'] = float((seg[-1]-seg[ip])/((len(seg)-1-ip)/FS)) if (len(seg)-1-ip) > 0 else np.nan
        o['up_down_ratio'] = (abs(o['up_slope'])/abs(o['down_slope'])
                              if o['down_slope'] not in (0, None) and np.isfinite(o['down_slope']) and abs(o['down_slope']) > 1e-9 else np.nan)
        o['up_arclen'] = float(np.sum(np.sqrt(1+np.diff(seg[:ip+1])**2))) if ip >= 1 else np.nan
        # enrichment D: sub-segments (thirds of the onset window) — slope + area per third
        L = len(seg); t1, t2 = L//3, 2*L//3
        for pidx, (lo_i, hi_i) in enumerate([(0, t1), (t1, t2), (t2, L-1)], start=1):
            sub = seg[lo_i:hi_i+1]
            o[f'seg_slope_p{pidx}'] = float((sub[-1]-sub[0])/((len(sub)-1)/FS)) if len(sub) > 1 else np.nan
            o[f'seg_area_p{pidx}']  = float(np.trapezoid(np.abs(sub-sub[0]))/FS) if len(sub) > 1 else np.nan
        # frequency texture of the QRS-delta window
        wfull = s[a:min(r+win40, len(s))]
        if wfull.size > 4:
            fftv = np.abs(np.fft.rfft(wfull-wfull.mean())); freqs = np.fft.rfftfreq(len(wfull), 1/FS); tot = fftv.sum()+1e-9
            o['freq_hf_ratio'] = float(fftv[(freqs >= 15) & (freqs <= 40)].sum()/tot)
            o['freq_hf_energy'] = float((fftv[(freqs >= 15) & (freqs <= 40)]**2).sum())
            csum = np.cumsum(fftv)/tot
            o['freq_median'] = float(freqs[np.searchsorted(csum, 0.5)]) if csum[-1] > 0 else np.nan
            o['freq_p95'] = float(freqs[min(np.searchsorted(csum, 0.95), len(freqs)-1)])
            o['freq_lf_ratio'] = float(fftv[(freqs >= 0.5) & (freqs < 5)].sum()/tot)
        o['R_amp'] = float(s[r])
        seg2 = s[a:min(r+win40, len(s))]
        if seg2.size: o['S_amp'] = float(seg2.min())
    # amplitudes anchored on delineation
    if Ron_i is not None and np.isfinite(Ron_i):
        on = int(Ron_i)
        if 0 <= on < r < len(s): o['Q_amp'] = float(s[on]); o['qrs_width_beat'] = (r-on)/FS*1000
    if Roff_i is not None and Tpk_i is not None and np.isfinite(Roff_i) and np.isfinite(Tpk_i):
        roff = int(Roff_i); tp = int(Tpk_i)
        if 0 <= roff < tp < len(s):
            o['st_level_J'] = float(s[roff]); o['st_slope'] = float((s[tp]-s[roff])/((tp-roff)/FS))
            o['t_peak_time'] = (tp-roff)/FS*1000; o['T_amp'] = float(s[tp])
            # enrichment: ST slope at J+40 / J+80 ms
            for ms, key in [(40,'J40'),(80,'J80')]:
                j = roff + int(ms/1000*FS)
                o[f'st_slope_{key}'] = float((s[min(j, len(s)-1)]-s[roff])/(ms/1000)) if j < len(s) else np.nan
            if Toff_i is not None and np.isfinite(Toff_i):
                toff = int(Toff_i)
                if tp < toff < len(s):
                    o['t_width'] = (toff-roff)/FS*1000
                    tseg = s[roff:toff]; o['t_area'] = float(np.trapezoid(np.abs(tseg-tseg[0]))/FS)
                    rise = tp-roff; fall = toff-tp; o['t_symmetry'] = rise/fall if fall > 0 else np.nan
                    # enrichment: T up/down slopes, peak-to-end
                    o['t_upslope']   = float((s[tp]-s[roff])/((tp-roff)/FS)) if (tp-roff) > 0 else np.nan
                    o['t_downslope'] = float((s[toff]-s[tp])/((toff-tp)/FS)) if (toff-tp) > 0 else np.nan
                    o['t_peak_to_end'] = (toff-tp)/FS*1000
    return o


def _discovery(s, rp, w, suf):
    """Per-beat morphology, summarized into med/std/IQR/skew per base (the enrichment)."""
    if rp is None:
        return {}
    rp = np.asarray(rp, int)
    Ron = np.asarray(w.get('ECG_R_Onsets', []), float); Roff = np.asarray(w.get('ECG_R_Offsets', []), float)
    Tpk = np.asarray(w.get('ECG_T_Peaks', []), float); Toff = np.asarray(w.get('ECG_T_Offsets', []), float)
    acc = {}
    for i, r in enumerate(rp):
        bm = _beat_morpho(s, r,
                          Ron[i] if i < len(Ron) else None, Roff[i] if i < len(Roff) else None,
                          Tpk[i] if i < len(Tpk) else None, Toff[i] if i < len(Toff) else None)
        for k, v in bm.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for base, vals in acc.items():
        sm = _summarize(vals)
        for stat in _SUMM:
            out[f'{base}_{stat}_{suf}'] = sm[stat]
    return out


def _process_one(m):
    warnings.filterwarnings('ignore')
    row = {'ecg_id': m['ecg_id'], 'patient_id': m['patient_id'], 'label': m['label'],
           'fold': m['fold'], 'source': m['source'], 'extraction_failed': 0}
    try:
        sig = load_signal(m['ecg_id'], m['source'])
        filt = {L: _bp(sig[:, LEAD_IDX[L]]) for L in LEADS_M1}
        delin = {L: _delineate(filt[L]) for L in LEADS_M1}
        clin, failed = _clinical(filt['II']); row.update(clin); row['extraction_failed'] = int(failed)
        nfail = 0
        for L in LEADS_M1:
            rp, w = delin[L]
            if rp is None: nfail += 1
            row.update(_discovery(filt[L], rp, w, L))
        row['n_leads_morpho_failed'] = nfail
    except Exception:
        row['extraction_failed'] = 1; row['n_leads_morpho_failed'] = 3
    return row


@contextlib.contextmanager
def _tqdm_joblib(t):
    import joblib
    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **k): t.update(n=self.batch_size); return super().__call__(*a, **k)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try:
        yield t
    finally:
        joblib.parallel.BatchCompletionCallBack = old; t.close()


def build_m1_features(meta, out_csv, force=False, n_jobs=10):
    """
    Build (or load) the ENRICHED M1 feature table. Same signature/guard as the canonical extractor.
    The DOC notebook points `out_csv` at the archived `m1_features_enriched.csv`; if it exists and not
    `force`, it is loaded (extraction skipped). NOTE: because column sets are alignment-sensitive, the
    DOC analysis should run off the archived CSV; a from-scratch rebuild reproduces the same feature
    families but is not guaranteed byte-identical to the archived table.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    if os.path.exists(out_csv) and not force:
        df = pd.read_csv(out_csv, dtype={'ecg_id': str})
        print(f"{os.path.basename(out_csv)} exists -> SKIPPED extraction (loaded {df.shape[0]}x{df.shape[1]}).")
        return df
    recs = meta.to_dict('records')
    sample = recs[:30]
    t0 = time.time()
    test_rows = [_process_one(m) for m in sample]
    nfeat = len([c for c in test_rows[0] if c not in
                 ('ecg_id','patient_id','label','fold','source','extraction_failed','n_leads_morpho_failed')])
    serial_min = (time.time()-t0)/len(sample)*len(recs)/60
    print(f"[pre-flight] {len(sample)} ECGs OK | {nfeat} ENRICHED features/ECG | "
          f"rough parallel ETA ~{serial_min/n_jobs:.0f}-{serial_min/4:.0f} min on {n_jobs} cores "
          f"(the progress bar shows the real one)")
    t0 = time.time()
    with _tqdm_joblib(tqdm(total=len(recs), desc='M1 enriched extraction', unit='ecg')):
        rows = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_process_one)(m) for m in recs)
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f"Built {os.path.basename(out_csv)} in {(time.time()