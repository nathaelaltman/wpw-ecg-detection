"""
fp_miss_code_audit.py -- repairs Section 7.5: a reproducible, committed-code audit of the
committee's 25 false positives (11 PTB-XL + 14 CSN) and 14 PTB-XL misses.

For every FP and every PTB-XL miss it dumps the FULL diagnostic-code list:
  - PTB-XL: scp_codes (acronym : likelihood) from ptbxl_database.csv
  - CSN:    SNOMED codes from the record .hea #Dx field, decoded to readable names via the
            committed data/raw/ningbo/.../ConditionNames_SNOMED-CT.csv

Then it produces the DEFINITIVE pre-excitation denominator for CSN:
  - PREEXC_SNOMED = every SNOMED code in CSN that denotes ventricular pre-excitation or an
    accessory-pathway re-entrant tachycardia, taken straight from ConditionNames:
        195060002  VPE   ventricular preexcitation
        74390002   WPW   WPW                          (the label code)
        233897008  AVRT  atrioventricular reentrant tachycardia (accessory-pathway mediated)
    (233896004 AVNRT is listed separately below for transparency: AVNRT is NODAL re-entry,
     NOT accessory-pathway, so it is reported but EXCLUDED from the pre-excitation set by default.)
  - how many CSN records carry ANY pre-excitation code (total, and per code)
  - exactly how many of the 14 CSN FP carry one, WITH the list
  => yields the reproducible line "X of the 25 FP are documented pre-excitation".

Read-only. Fast (grep-style scan of .hea headers). fold 10 untouched.

Run:  python fp_miss_code_audit.py
Writes: reports/metrics/fp_miss_code_audit.json
"""
import os, sys, json, ast, glob, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
ENS  = os.path.join(ROOT, "models", "ensemble")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
sys.path.insert(0, SRC)
from signal_loading import resolve_path, NINGBO_ROOT

THR_ENS = 0.9969; ALPHA = 0.5

# pre-excitation code set (accessory-pathway); AVNRT reported but excluded by default
PREEXC = {"195060002": "VPE ventricular preexcitation",
          "74390002":  "WPW",
          "233897008": "AVRT atrioventricular reentrant tachycardia"}
# NOTE (2026-07-24): the AVNRT code was 233896004, taken from the corpus ConditionNames file.
# That code appears NOWHERE in the corpus. The SNOMED code for atrioventricular NODAL re-entrant
# tachycardia is 251166008 (PhysioNet/CinC 2021 mapping), present 16 times, all in negatives.
# The reported AVNRT count was therefore misleadingly zero. No effect on the pre-excitation set
# or on any paper number: AVNRT is nodal re-entry, not an accessory pathway, so it was excluded
# from PREEXC by construction either way.
AVNRT  = {"251166008": "AVNRT (nodal reentry - NOT accessory pathway; reported, excluded)"}

# ---- committee reconstruction (frozen recipe) ----
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

FP  = M[(M.label==0) & (M.pred==1)]
FN  = M[(M.label==1) & (M.pred==0)]
FP_ptb = FP[FP.source.map(is_ptb)]; FP_csn = FP[~FP.source.map(is_ptb)]
FN_ptb = FN[FN.source.map(is_ptb)]
print("committee: FP=%d (ptb %d / csn %d) | FN_ptb=%d (expected 25=11+14 ; 14)" % (
    len(FP), len(FP_ptb), len(FP_csn), len(FN_ptb)))

# ---- PTB-XL scp_codes ----
ptb_db = None
for r, _, files in os.walk(os.path.join(ROOT, "data", "raw", "ptbxl")):
    if "ptbxl_database.csv" in files: ptb_db = os.path.join(r, "ptbxl_database.csv")
DB = pd.read_csv(ptb_db)[["ecg_id","scp_codes"]]; DB["ecg_id"] = DB["ecg_id"].astype(int).astype(str)
SCP = {}
for _, r in DB.iterrows():
    try: SCP[r.ecg_id] = ast.literal_eval(r.scp_codes)
    except Exception: SCP[r.ecg_id] = {}

# ---- CSN readable SNOMED names ----
cn_path = None
for r, _, files in os.walk(os.path.join(ROOT, "data", "raw", "ningbo")):
    if "ConditionNames_SNOMED-CT.csv" in files: cn_path = os.path.join(r, "ConditionNames_SNOMED-CT.csv")
CN = pd.read_csv(cn_path)
SNOMED_NAME = {str(row["Snomed_CT"]): f'{row["Acronym Name"]} ({str(row["Full Name"]).strip()})'
               for _, row in CN.iterrows()}

def csn_dx(ecg_id, source):
    """Read #Dx SNOMED list from the record .hea header."""
    try: p = resolve_path(ecg_id, source) + ".hea"
    except Exception: return []
    if not os.path.exists(p): return []
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.startswith("#Dx"):
            codes = line.split(":", 1)[1].strip()
            return [c.strip() for c in re.split(r"[,\s]+", codes) if c.strip()]
    return []

def dump_ptb(df):
    out = []
    for _, r in df.iterrows():
        codes = {k: v for k, v in SCP.get(str(r.ecg_id), {}).items()}
        out.append(dict(ecg_id=str(r.ecg_id), source=r.source, fold=int(r.fold),
                        ens=round(float(r.ens),4), scp_codes=codes))
    return out

def dump_csn(df):
    out = []
    for _, r in df.iterrows():
        codes = csn_dx(r.ecg_id, r.source)
        named = [{"snomed": c, "name": SNOMED_NAME.get(c, "UNKNOWN")} for c in codes]
        out.append(dict(ecg_id=str(r.ecg_id), source=r.source, fold=int(r.fold),
                        ens=round(float(r.ens),4), dx=named,
                        preexc=[c for c in codes if c in PREEXC]))
    return out

fp_ptb = dump_ptb(FP_ptb); fp_csn = dump_csn(FP_csn); fn_ptb = dump_ptb(FN_ptb)

# ---- enumerate ALL pre-excitation carriers across the whole CSN corpus ----
def all_csn_headers():
    return glob.glob(os.path.join(str(NINGBO_ROOT), "**", "*.hea"), recursive=True)

def scan_preexc_corpus():
    counts = {c: 0 for c in list(PREEXC) + list(AVNRT)}; carriers = {c: [] for c in counts}
    any_carriers = set()
    for p in all_csn_headers():
        rid = os.path.splitext(os.path.basename(p))[0]
        for line in open(p, encoding="utf-8", errors="ignore"):
            if line.startswith("#Dx"):
                codes = set(re.split(r"[,\s]+", line.split(":",1)[1].strip()))
                for c in counts:
                    if c in codes:
                        counts[c] += 1; carriers[c].append(rid)
                        if c in PREEXC: any_carriers.add(rid)
                break
    return counts, carriers, sorted(any_carriers)

print("scanning CSN headers for pre-excitation codes (a few seconds)...")
pe_counts, pe_carriers, pe_any = scan_preexc_corpus()

# how many of the 14 CSN FP carry a pre-excitation code
fp_csn_preexc = [d for d in fp_csn if d["preexc"]]

print("\n=== CSN pre-excitation code carriers (whole corpus) ===")
for c in PREEXC:
    print("  %-11s %-45s %d records" % (c, PREEXC[c], pe_counts[c]))
for c in AVNRT:
    print("  %-11s %-45s %d records  [excluded]" % (c, AVNRT[c], pe_counts[c]))
print("  ANY accessory-pathway pre-excitation code: %d records" % len(pe_any))

print("\n=== the 14 CSN FP: how many are documented pre-excitation ===")
print("  %d of the 14 CSN FP carry a pre-excitation code:" % len(fp_csn_preexc))
for d in fp_csn_preexc:
    names = ", ".join(SNOMED_NAME.get(c, c) for c in d["preexc"])
    print("    %s  ens=%.4f  -> %s" % (d["ecg_id"], d["ens"], names))

# denominator line
tot_fp = len(FP)
print("\n=== reproducible denominator ===")
print("  Of the %d committee false positives, %d are CSN records carrying a documented" % (tot_fp, len(fp_csn_preexc)))
print("  ventricular-pre-excitation / accessory-pathway SNOMED code (label code 74390002")
print("  excluded from the positive class only by the likelihood/definition rule).")

out = dict(
    committee=dict(FP_total=tot_fp, FP_ptb=len(FP_ptb), FP_csn=len(FP_csn), FN_ptb=len(FN_ptb)),
    preexc_code_set=PREEXC, avnrt_excluded=AVNRT,
    csn_preexc_corpus_counts=pe_counts, csn_preexc_carriers=pe_carriers,
    csn_preexc_any_total=len(pe_any), csn_preexc_any_ids=pe_any,
    fp_csn_preexc_count=len(fp_csn_preexc), fp_csn_preexc=fp_csn_preexc,
    FP_ptb=fp_ptb, FP_csn=fp_csn, FN_ptb=fn_ptb)
json.dump(out, open(os.path.join(MET, "fp_miss_code_audit.json"), "w"), indent=2, default=str)
print("\nWrote reports/metrics/fp_miss_code_audit.json")
