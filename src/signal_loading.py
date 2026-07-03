"""
signal_loading.py — Canonical ECG signal loader (PTB-XL + Ningbo).
Single source of truth for reading a signal anywhere in the project (M1-M5, M7, Flask).
Guarantees: shape (5000, 12), 500 Hz, mV, standard uppercase lead order.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root (contains config.py)
from config import PROCESSED, RAW

PTBXL_BASE  = RAW / "ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
NINGBO_ROOT = RAW / "ningbo/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"

# Canonical order of the 12 leads (uppercase). The WHOLE project aligns on this.
LEADS_CANONICAL = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

# ecg_id -> rel_path lookup for Ningbo (loaded once on first call).
_ningbo_path = None
def _ningbo_lookup():
    global _ningbo_path
    if _ningbo_path is None:
        n = pd.read_csv(PROCESSED / "metadata_ningbo.csv", dtype={'ecg_id': str})
        _ningbo_path = dict(zip(n['ecg_id'], n['rel_path']))
    return _ningbo_path


def resolve_path(ecg_id, source):
    """Absolute wfdb path WITHOUT extension, depending on the dataset."""
    s = str(source).lower()
    if 'ptb' in s:
        k = int(ecg_id)
        folder = f"{(k // 1000) * 1000:05d}"   # 1->00000, 1000->01000
        return str(PTBXL_BASE / "records500" / folder / f"{k:05d}_hr")
    rp = _ningbo_lookup()[str(ecg_id)]
    p = NINGBO_ROOT / rp
    return str(p.with_suffix('')) if p.suffix else str(p)


def load_signal(ecg_id, source, check=True):
    """
    Load an ECG -> ndarray (5000, 12) in mV, leads in LEADS_CANONICAL order.
    Reorders columns if needed (robust to a different on-disk order).
    check=True: verifies shape/fs and raises an explicit error on any anomaly.
    """
    path = resolve_path(ecg_id, source)
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal                      # (n_samples, n_leads), physical units (mV)
    names = [str(n).upper() for n in rec.sig_name]

    # Reorder columns to the canonical order (handles case AND order).
    if names != LEADS_CANONICAL:
        idx = []
        for lead in LEADS_CANONICAL:
            if lead not in names:
                raise ValueError(f"{source}/{ecg_id}: lead '{lead}' missing. Seen: {names}")
            idx.append(names.index(lead))
        sig = sig[:, idx]

    if check:
        if sig.shape != (5000, 12):
            raise ValueError(f"{source}/{ecg_id}: shape {sig.shape} != (5000,12)")
        if rec.fs != 500:
            raise ValueError(f"{source}/{ecg_id}: fs {rec.fs} != 500 Hz")

    return sig.astype(np.float32)
