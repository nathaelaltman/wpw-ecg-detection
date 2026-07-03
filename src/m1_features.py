"""
m1_features.py — NeuroKit2 clinical + morphology feature extraction for M1 (leads II/V1/V5).

Single source of truth for the M1 feature table, imported by the M1 notebooks (05_m1_clinical_*) (extraction defined ONCE).
Applies the frozen 0.5-40 Hz filter. ~172 per-lead clinical+morphology/delta features.

Thesis: the WPW delta wave destabilizes NeuroKit delineation (R_Onset/P_onset mis-placed), so the
clinical intervals become non-discriminant; the signal lives in form/delta features that survive the
broken delineation. NaNs are kept (extraction failure is itself a signal; XGBoost handles them).

Note: an enriched pool (per-beat distribution summaries + sub-segments, ~700 features) was tested in
the DOCUMENTATION notebook `05_m1_enrichment_rejected` (extractor `m1_features_enriched.py`).
It did NOT lift M1's ceiling (signal-limited) -> this simpler pool is the retained canonical one.

Main entry point:
    build_m1_features(meta, out_csv, force=False) -> pandas.DataFrame
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


def _med(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.median(v)) if v.size else np.nan
def _std(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.std(v)) if v.size >= 2 else np.nan
def _cv(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.std(v)/abs(np.mean(v))) if v.size >= 2 and abs(np.mean(v)) > 1e-9 else np.nan


def _delineate(s):
    import neurokit2 as nk
    _, info = nk.ecg_peaks(s, sampling_rate=FS); rp = info["ECG_R_Peaks"]
    if rp is None or len(rp) < 3:
        return None, None
    _, w = nk.ecg_delineate(s, rpeaks=rp, sampling_rate=FS, method="dwt")
    return np.asarray(rp), w


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


DISC_BASES = (
    ['delta_slopemax20','delta_slopemax40','delta_slopemax60','delta_slopemean','delta_t_to_max',
     'delta_ratio_init_peak','delta_accel_max','delta_empatement_area','delta_slow_phase_ms',
     'delta_vel_quarter','delta_vel_half','delta_area_third']
    + ['qrs_area','qrs_absarea','qrs_energy','qrs_ninflect','qrs_nzero','qrs_skew','qrs_kurt','qrs_p2p',
       'qrs_fwhm','qrs_asym','qrs_arclen','qrs_posneg_ratio','qrs_npeaks','qrs_moment3','qrs_entropy']
    + ['R_amp','S_amp','Q_amp','T_amp','RS_ratio','QR_ratio','TR_ratio']
    + ['st_slope','st_level_J','t_area','t_width','t_peak_time','t_symmetry','st_t_area_ratio']
    + ['freq_hf_ratio','freq_hf_energy','freq_median','freq_p95','freq_lf_ratio']
    + ['var_qrswidth_std','var_qrswidth_cv','var_Ramp_std','var_Ramp_cv','var_delta_std','var_delta_cv','var_area_std']
)

def _discovery(s, rp, w, suf):
    o = {f'{b}_{suf}': np.nan for b in DISC_BASES}
    if rp is None:
        return o
    rp = np.asarray(rp, int); Ron = np.asarray(w.get('ECG_R_Onsets', []), float)
    Roff = np.asarray(w.get('ECG_R_Offsets', []), float); Tpk = np.asarray(w.get('ECG_T_Peaks', []), float)
    Toff = np.asarray(w.get('ECG_T_Offsets', []), float)
    win40 = int(0.040*FS)
    acc = {k: [] for k in DISC_BASES}
    qw_bb, ramp_bb, delta_bb, area_bb = [], [], [], []
    for i, r in enumerate(rp):
        a = max(0, r-win40)
        if r > a+3:
            seg = s[a:r+1]; d1 = np.diff(seg)*FS; absd = np.abs(d1)
            for ms, key in [(20,'20'),(40,'40'),(60,'60')]:
                npt = int(ms/1000*FS); segk = s[max(0, r-npt):r+1]
                if segk.size > 2: acc[f'delta_slopemax{key}'].append(float(np.max(np.abs(np.diff(segk)*FS))))
            acc['delta_slopemean'].append(float(absd.mean()))
            imax = int(np.argmax(absd)); acc['delta_t_to_max'].append(imax/FS*1000)
            init = np.mean(absd[:max(1, len(absd)//3)]); pk = absd.max()
            acc['delta_ratio_init_peak'].append(float(init/pk) if pk > 1e-9 else np.nan)
            d2 = np.diff(d1)*FS; acc['delta_accel_max'].append(float(np.max(np.abs(d2))) if len(d2) else np.nan)
            ideal = np.linspace(seg[0], seg[-1], len(seg)); acc['delta_empatement_area'].append(float(np.trapezoid(np.abs(seg-ideal))/FS))
            thr = 0.5*pk; idx = np.where(absd >= thr)[0]; acc['delta_slow_phase_ms'].append(float(idx[0]/FS*1000) if idx.size else np.nan)
            q = len(seg)//4; h = len(seg)//2
            acc['delta_vel_quarter'].append(float((seg[q]-seg[0])/(q/FS)) if q > 0 else np.nan)
            acc['delta_vel_half'].append(float((seg[h]-seg[0])/(h/FS)) if h > 0 else np.nan)
            third = seg[:max(2, len(seg)//3)]; acc['delta_area_third'].append(float(np.trapezoid(np.abs(third-third[0]))/FS))
            rng = seg.max()-seg.min()
            acc['qrs_area'].append(float(np.trapezoid(seg-seg[0])/FS)); acc['qrs_absarea'].append(float(np.trapezoid(np.abs(seg-seg[0]))/FS))
            acc['qrs_energy'].append(float(np.sum(seg**2)))
            dd2 = np.diff(np.sign(np.diff(seg))); acc['qrs_ninflect'].append(int(np.sum(dd2 != 0)))
            acc['qrs_nzero'].append(int(np.sum(np.diff(np.sign(seg-seg.mean())) != 0)))
            if seg.size > 3: acc['qrs_skew'].append(float(skew(seg))); acc['qrs_kurt'].append(float(kurtosis(seg)))
            acc['qrs_p2p'].append(float(rng))
            half = seg.min()+rng/2; above = np.where(seg >= half)[0]
            acc['qrs_fwhm'].append(float((above[-1]-above[0])/FS*1000) if above.size > 1 else np.nan)
            pkp = int(np.argmax(np.abs(seg-seg[0]))); acc['qrs_asym'].append(pkp/(len(seg)-pkp) if len(seg)-pkp > 0 else np.nan)
            acc['qrs_arclen'].append(float(np.sum(np.sqrt(1+np.diff(seg)**2))))
            pos = seg[seg > seg.mean()].sum(); negv = abs(seg[seg < seg.mean()].sum())
            acc['qrs_posneg_ratio'].append(float(pos/negv) if negv > 1e-9 else np.nan)
            ddp = np.diff(np.sign(np.diff(seg))); acc['qrs_npeaks'].append(int(np.sum(ddp < 0)))
            acc['qrs_moment3'].append(float(np.mean((seg-seg.mean())**3)))
            pp = np.abs(seg-seg.min())+1e-9; pp = pp/pp.sum(); acc['qrs_entropy'].append(float(-np.sum(pp*np.log(pp))))
            wfull = s[a:min(r+win40, len(s))]
            if wfull.size > 4:
                fftv = np.abs(np.fft.rfft(wfull-wfull.mean())); freqs = np.fft.rfftfreq(len(wfull), 1/FS); tot = fftv.sum()+1e-9
                acc['freq_hf_ratio'].append(float(fftv[(freqs >= 15) & (freqs <= 40)].sum()/tot))
                acc['freq_hf_energy'].append(float((fftv[(freqs >= 15) & (freqs <= 40)]**2).sum()))
                csum = np.cumsum(fftv)/tot
                acc['freq_median'].append(float(freqs[np.searchsorted(csum, 0.5)]) if csum[-1] > 0 else np.nan)
                acc['freq_p95'].append(float(freqs[min(np.searchsorted(csum, 0.95), len(freqs)-1)]))
                acc['freq_lf_ratio'].append(float(fftv[(freqs >= 0.5) & (freqs < 5)].sum()/tot))
            acc['R_amp'].append(float(s[r])); ramp_bb.append(float(s[r]))
            seg2 = s[a:min(r+win40, len(s))]
            if seg2.size: acc['S_amp'].append(float(seg2.min()))
            if acc['delta_slopemax40']: delta_bb.append(acc['delta_slopemax40'][-1])
            area_bb.append(acc['qrs_area'][-1])
        if i < len(Ron) and np.isfinite(Ron[i]):
            on = int(Ron[i])
            if 0 <= on < r < len(s): acc['Q_amp'].append(float(s[on])); qw_bb.append((r-on)/FS*1000)
        if i < len(Roff) and i < len(Tpk) and np.isfinite(Roff[i]) and np.isfinite(Tpk[i]):
            roff = int(Roff[i]); tp = int(Tpk[i])
            if 0 <= roff < tp < len(s):
                acc['st_level_J'].append(float(s[roff])); acc['st_slope'].append(float((s[tp]-s[roff])/((tp-roff)/FS)))
                acc['t_peak_time'].append((tp-roff)/FS*1000); acc['T_amp'].append(float(s[tp]))
                if i < len(Toff) and np.isfinite(Toff[i]):
                    toff = int(Toff[i])
                    if tp < toff < len(s):
                        acc['t_width'].append((toff-roff)/FS*1000)
                        tseg = s[roff:toff]; acc['t_area'].append(float(np.trapezoid(np.abs(tseg-tseg[0]))/FS))
                        rise = tp-roff; fall = toff-tp; acc['t_symmetry'].append(rise/fall if fall > 0 else np.nan)
    for k in DISC_BASES:
        if k.startswith('var_') or k in ('RS_ratio','QR_ratio','TR_ratio','st_t_area_ratio'):
            continue
        o[f'{k}_{suf}'] = _med(acc[k]) if acc.get(k) else np.nan
    ra = _med(acc['R_amp']); sa = _med(acc['S_amp']); qa = _med(acc['Q_amp']); ta = _med(acc['T_amp'])
    o[f'RS_ratio_{suf}'] = ra/abs(sa) if sa and abs(sa) > 1e-6 else np.nan
    o[f'QR_ratio_{suf}'] = qa/ra if ra and abs(ra) > 1e-6 else np.nan
    o[f'TR_ratio_{suf}'] = ta/ra if ra and abs(ra) > 1e-6 else np.nan
    sl = _med(acc['st_slope']); tar = _med(acc['t_area'])
    o[f'st_t_area_ratio_{suf}'] = sl/tar if tar and abs(tar) > 1e-6 else np.nan
    o[f'var_qrswidth_std_{suf}'] = _std(qw_bb); o[f'var_qrswidth_cv_{suf}'] = _cv(qw_bb)
    o[f'var_Ramp_std_{suf}'] = _std(ramp_bb); o[f'var_Ramp_cv_{suf}'] = _cv(ramp_bb)
    o[f'var_delta_std_{suf}'] = _std(delta_bb); o[f'var_delta_cv_{suf}'] = _cv(delta_bb)
    o[f'var_area_std_{suf}'] = _std(area_bb)
    return o


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
    Build (or load) the M1 feature table for all ECGs in `meta`.
    Guarded: if `out_csv` exists and not `force`, it is loaded instead of recomputed.
    A short pre-flight test (first ~30 records) runs before the full extraction to catch bugs fast.
    Returns the DataFrame.
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
    print(f"[pre-flight] {len(sample)} ECGs OK | {nfeat} features/ECG | "
          f"rough parallel ETA ~{serial_min/n_jobs:.0f}-{serial_min/4:.0f} min on {n_jobs} cores "
          f"(the progress bar shows the real one)")
    t0 = time.time()
    with _tqdm_joblib(tqdm(total=len(recs), desc='M1 extraction', unit='ecg')):
        rows = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_process_one)(m) for m in recs)
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f"Built {os.path.basename(out_csv)} in {(time.time()