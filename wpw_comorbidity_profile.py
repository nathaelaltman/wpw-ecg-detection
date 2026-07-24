"""
wpw_comorbidity_profile.py -- descriptive demographic and diagnostic profile of the FULL cohort
(66,951 ECGs: PTB-XL 21,799 + Chapman-Shaoxing-Ningbo 45,152), split WPW vs non-WPW, for
Section 3 of the paper.

PURELY DESCRIPTIVE. No test, no model, no threshold. Read-only.

WHAT IT PRODUCES, PER CORPUS AND NEVER POOLED
The two corpora use different vocabularies (SCP-ECG statements for PTB-XL, SNOMED-CT for CSN).
Merging them would require an arbitrary mapping that a reviewer can attack, so every corpus
keeps its own block with its own denominators.

  1. Demographics per corpus and per class: n, age median/IQR/range, sex split, missingness.
  2. Every diagnostic code in the corpus, with its count and percent among WPW records AND
     among non-WPW records. The contrast column is the point: it shows whether the WPW records
     carry a different diagnostic mix than the surrounding archive.
  3. Diagnostic BURDEN: distribution of the number of non-pre-excitation codes per record
     (0, 1, 2, 3+), WPW vs non-WPW. This is vocabulary-neutral, so it IS comparable across the
     two corpora, and it is the denominator Section 7.3 (comorbidity masking) actually needs:
     how many WPW records are isolated, with nothing else on the tracing.
  4. A ready-to-paste LaTeX table in reports/metrics/wpw_comorbidity_table.tex

NO STATISTICAL TEST IS PERFORMED. The table describes; it does not infer. Section 7.3 carries
the only inferential claim about comorbidity, and it is Holm-corrected there.

THREE TRAPS HANDLED EXPLICITLY (each would silently corrupt the counts)

  (a) PTB-XL scp_codes likelihood 0.0 does NOT mean absent. Per the PTB-XL documentation a
      likelihood of 0 means the statement is present but no likelihood was assigned by the
      annotator. Counting only likelihood > 0 would drop most statements. A code counts as
      PRESENT if the key exists, whatever its likelihood; the number of zero-likelihood
      carriers is reported separately so the choice stays auditable.
  (b) PTB-XL censors age above 89 by storing 300. Recoded to NaN for median/IQR, counted
      separately as "age > 89 (censored)".
  (c) Scanning 45k CSN headers is the only slow step, so it is cached to
      reports/metrics/_csn_dx_cache.csv and reloaded on re-run (delete the file to force a
      rescan). The cache is keyed on ecg_id, so it cannot drift from the cohort definition.

PRE-EXCITATION CODES ARE EXCLUDED from the burden counts and flagged in the code table, because
they describe the label itself rather than a concomitant condition:
  PTB-XL : WPW
  CSN    : 74390002 (WPW), 195060002 (VPE, ventricular pre-excitation),
           233897008 (AVRT, accessory-pathway re-entrant tachycardia)
CSN 233896004 (AVNRT) is NODAL re-entry, not accessory-pathway, so it is a genuine concomitant
diagnosis and stays in the counts.

WHAT THIS MEASURES: ECG-statement labels, not clinical histories. A record coded AFIB carries
that statement on the tracing; it does not certify a documented history of atrial fibrillation.
The caption says so.

fold 10 is included: describing the composition of the cohort is not evaluating a model on it.

Run:  python wpw_comorbidity_profile.py
Writes: reports/metrics/wpw_comorbidity_profile.json
        reports/metrics/wpw_comorbidity_table.tex
        reports/metrics/_csn_dx_cache.csv   (scan cache)
"""
import os, sys, json, ast, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
SRC  = os.path.join(ROOT, "src")
MET  = os.path.join(ROOT, "reports", "metrics"); os.makedirs(MET, exist_ok=True)
sys.path.insert(0, SRC)
from signal_loading import resolve_path

MIN_PCT_IN_TABLE = 1.0   # a code appears in the LaTeX table if it reaches this pct in EITHER class
ZERO_ROW_MIN_PCT = 3.0   # a code ABSENT from the WPW group is kept only if it reaches this pct
                         # among non-WPW: its absence is then a finding (e.g. "Normal ECG 0 vs
                         # 43.8%", "Atrial fibrillation 0/72 vs 4.0%"). Below it, a zero row is
                         # noise. Set to 0 to keep every zero row. These two constants reproduce
                         # the two tables printed in the paper (24 and 28 rows).
CACHE = os.path.join(MET, "_csn_dx_cache.csv")

PTB_PREEXC = {"WPW"}
CSN_PREEXC = {"74390002": "WPW",
              "195060002": "VPE ventricular pre-excitation",
              "233897008": "AVRT accessory-pathway re-entrant tachycardia"}

# Baseline rhythm / normality statements. These describe the underlying rhythm or the absence of
# a finding, NOT a concomitant disease. Counting them as comorbidity is what makes a WPW record
# coded {WPW, sinus rhythm} look like it carries one comorbidity when it carries none. They are
# therefore excluded from the SECOND burden metric ("pathological statements"), and the excluded
# set is printed and put in the caption so the choice is auditable. Pathological rhythms
# (atrial flutter, AF, tachyarrhythmias other than plain sinus) are deliberately NOT excluded.
PTB_BASELINE = {"NORM", "SR"}
CSN_BASELINE = {"426783006",   # sinus rhythm
                "426177001",   # sinus bradycardia
                "427084000",   # sinus tachycardia
                "427393009"}   # sinus irregularity / arrhythmia

# SNOMED names absent from the dataset's own ConditionNames_SNOMED-CT.csv (that file covers the
# Chapman-Shaoxing half; Ningbo introduced further codes). Filled from the official PhysioNet/
# Computing in Cardiology Challenge 2021 mapping (dx_mapping_scored.csv + dx_mapping_unscored.csv,
# github.com/physionetchallenges/evaluation-2021). Used only as a FALLBACK: the dataset's own
# names win when present, so nothing already resolved changes.
PHYSIONET_2021 = {
    "164889003": "AF (atrial fibrillation)", "164890007": "AFL (atrial flutter)",
    "6374002": "BBB (bundle branch block)", "426627000": "Brady (bradycardia)",
    "733534002": "CLBBB (complete left bundle branch block)",
    "713427006": "CRBBB (complete right bundle branch block)",
    "270492004": "IAVB (1st degree av block)",
    "713426002": "IRBBB (incomplete right bundle branch block)",
    "39732003": "LAD (left axis deviation)", "445118002": "LAnFB (left anterior fascicular block)",
    "164909002": "LBBB (left bundle branch block)", "251146004": "LQRSV (low qrs voltages)",
    "698252002": "NSIVCB (nonspecific intraventricular conduction disorder)",
    "426783006": "NSR (sinus rhythm)", "284470004": "PAC (premature atrial contraction)",
    "10370003": "PR (pacing rhythm)", "365413008": "PRWP (poor R wave progression)",
    "427172004": "PVC (premature ventricular contractions)",
    "164947007": "LPR (prolonged pr interval)", "111975006": "LQT (prolonged qt interval)",
    "164917005": "QAb (qwave abnormal)", "47665007": "RAD (right axis deviation)",
    "59118001": "RBBB (right bundle branch block)", "427393009": "SA (sinus arrhythmia)",
    "426177001": "SB (sinus bradycardia)", "427084000": "STach (sinus tachycardia)",
    "63593006": "SVPB (supraventricular premature beats)", "164934002": "TAb (t wave abnormal)",
    "59931005": "TInv (t wave inversion)", "17338001": "VPB (ventricular premature beats)",
    "233892002": "AAR (accelerated atrial escape rhythm)", "164951009": "abQRS (abnormal QRS)",
    "251187003": "AED (atrial escape beat)", "61277005": "AIVR (accelerated idioventricular rhythm)",
    "426664006": "AJR (accelerated junctional rhythm)",
    "251139008": "ALR (suspect arm ecg leads reversed)",
    "57054005": "AMI (acute myocardial infarction)", "413444003": "AMIs (acute myocardial ischemia)",
    "426434006": "AnMIs (anterior ischemia)", "54329005": "AnMI (anterior myocardial infarction)",
    "251173003": "AB (atrial bigeminy)", "195080001": "AFAFL (atrial fibrillation and flutter)",
    "195126007": "AH (atrial hypertrophy)", "251268003": "AP (atrial pacing pattern)",
    "106068003": "ARH (atrial rhythm)", "713422000": "ATach (atrial tachycardia)",
    "233917008": "AVB (av block)", "50799005": "AVD (atrioventricular dissociation)",
    "29320008": "AVJR (atrioventricular junctional rhythm)",
    "251166008": "AVNRT (atrioventricular node reentrant tachycardia)",
    "233897008": "AVRT (atrioventricular reentrant tachycardia)",
    "251170000": "BPAC (blocked premature atrial contraction)", "418818005": "BRU (brugada)",
    "74615001": "BTS (brady tachy syndrome)", "426749004": "CAF (chronic atrial fibrillation)",
    "251199005": "CCR (counterclockwise rotation)",
    "61721007": "CVCL/CCVCL (clockwise or counterclockwise vectorcardiographic loop)",
    "698247007": "CD (cardiac dysrhythmia)", "27885002": "CHB (complete heart block)",
    "204384007": "CIAHB (congenital incomplete atrioventricular heart block)",
    "53741008": "CHD (coronary heart disease)", "413844008": "CMI (chronic myocardial ischemia)",
    "251198002": "CR (clockwise rotation)", "82226007": "DIB (diffuse intraventricular block)",
    "428417006": "ERe (early repolarization)", "13640000": "FB (fusion beats)",
    "164942001": "FQRS (fqrs wave)", "84114007": "HF (heart failure)",
    "368009": "HVD (heart valve disorder)", "251259000": "HTV (high t-voltage)",
    "251200008": "ICA (indeterminate cardiac axis)", "195042002": "IIAVB (2nd degree av block)",
    "426183003": "IIAVBII (mobitz type II atrioventricular block)",
    "425419005": "IIs (inferior ischaemia)",
    "251120003": "ILBBB (incomplete left bundle branch block)",
    "704997005": "ISTD (inferior ST segment depression)", "49260003": "IR (idioventricular rhythm)",
    "426995002": "JE (junctional escape)", "251164006": "JPC (junctional premature complex)",
    "426648003": "JTach (junctional tachycardia)", "253352002": "LAA (left atrial abnormality)",
    "67741000119109": "LAE (left atrial enlargement)", "446813000": "LAH (left atrial hypertrophy)",
    "425623009": "LIs (lateral ischaemia)", "445211001": "LPFB (left posterior fascicular block)",
    "164873001": "LVH (left ventricular hypertrophy)",
    "55827005": "LVHV (left ventricular high voltage)", "370365005": "LVS (left ventricular strain)",
    "164865005": "MI (myocardial infarction)", "164861001": "MIs (myocardial ischemia)",
    "54016002": "MoI (mobitz type i wenckebach atrioventricular block)",
    "428750005": "NSSTTA (nonspecific st t abnormality)",
    "164867002": "OldMI (old myocardial infarction)",
    "282825002": "PAF (paroxysmal atrial fibrillation)", "251205003": "PPW (prolonged P wave)",
    "67198005": "PSVT (paroxysmal supraventricular tachycardia)",
    "425856008": "PVT (paroxysmal ventricular tachycardia)", "164912004": "PWC (p wave change)",
    "253339007": "RAAb (right atrial abnormality)", "164921003": "RAb (r wave abnormal)",
    "446358003": "RAH (right atrial hypertrophy)",
    "67751000119106": "RAHV (right atrial high voltage)",
    "314208002": "RAF (rapid atrial fibrillation)",
    "89792004": "RVH (right ventricular hypertrophy)",
    "17366009": "SAAWR (sinus atrium to atrial wandering rhythm)",
    "65778007": "SAB (sinoatrial block)", "5609005": "SARR (sinus arrest)",
    "60423000": "SND (sinus node dysfunction)", "49578007": "SPRI (shortened pr interval)",
    "77867006": "SQT (decreased qt interval)", "55930002": "STC (s t changes)",
    "429622005": "STD (st depression)", "164931005": "STE (st elevation)",
    "164930006": "STIAb (st interval abnormal)", "251168009": "SVB (supraventricular bigeminy)",
    "426761007": "SVT (supraventricular tachycardia)",
    "266257000": "TIA (transient ischemic attack)", "251223006": "TPW (tall p wave)",
    "164937009": "UAb (u wave abnormal)", "11157007": "VBig (ventricular bigeminy)",
    "164884008": "VEB (ventricular ectopics)", "75532003": "VEsB (ventricular escape beat)",
    "81898007": "VEsR (ventricular escape rhythm)", "164896001": "VF (ventricular fibrillation)",
    "111288001": "VFL (ventricular flutter)", "266249003": "VH (ventricular hypertrophy)",
    "195060002": "VPEx (ventricular pre excitation)", "251266004": "VPP (ventricular pacing pattern)",
    "251182009": "VPVC (paired ventricular premature complexes)",
    "164895002": "VTach (ventricular tachycardia)", "251180001": "VTrig (ventricular trigeminy)",
    "195101003": "WAP (wandering atrial pacemaker)",
    "74390002": "WPW (wolff parkinson white pattern)"}


def find(root_sub, fname):
    for r, _, files in os.walk(os.path.join(ROOT, "data", "raw", root_sub)):
        if fname in files:
            return os.path.join(r, fname)
    raise FileNotFoundError(f"{fname} not found under data/raw/{root_sub}")


# ---------------------------------------------------------------- cohort
meta = pd.read_csv(os.path.join(PROC, "metadata_combined.csv"), dtype={"ecg_id": str})
meta["is_ptb"] = meta.source.astype(str).str.lower().str.contains("ptb")
PTB, CSN = meta[meta.is_ptb], meta[~meta.is_ptb]
print("cohort: %d ECGs = %d PTB-XL (%d WPW) + %d CSN (%d WPW)"
      % (len(meta), len(PTB), int(PTB.label.sum()), len(CSN), int(CSN.label.sum())))


# ---------------------------------------------------------------- PTB-XL: parse all records
db = pd.read_csv(find("ptbxl", "ptbxl_database.csv"))
db["ecg_id"] = db["ecg_id"].astype(int).astype(str)
db = db.set_index("ecg_id")

stmt = pd.read_csv(find("ptbxl", "scp_statements.csv"), index_col=0)
SCP_NAME = {str(i): str(r.get("description", i)) for i, r in stmt.iterrows()}

ptb_rows = []
missing_ptb = 0
for eid, lab in zip(PTB.ecg_id, PTB.label):
    if eid not in db.index:
        missing_ptb += 1; continue
    row = db.loc[eid]
    try:
        d = ast.literal_eval(row["scp_codes"])
    except Exception:
        d = {}
    codes = sorted(d.keys())
    comorb = [k for k in codes if k not in PTB_PREEXC]
    age = row.get("age", np.nan)
    ptb_rows.append(dict(ecg_id=eid, label=int(lab),
                         age=(float(age) if pd.notna(age) else None),
                         sex=(None if pd.isna(row.get("sex")) else int(row["sex"])),
                         codes=codes, zero_lik=[k for k in codes if float(d[k]) == 0.0],
                         n_comorb=len(comorb)))
if missing_ptb:
    print("  WARNING: %d PTB-XL ecg_id absent from ptbxl_database.csv" % missing_ptb)


# ---------------------------------------------------------------- CSN: scan headers (cached)
# NAMING PRIORITY: the SNOMED-CT preferred term (via the PhysioNet 2021 mapping) WINS; the
# corpus's own ConditionNames file is used only for codes the mapping does not cover.
# This is deliberate and was reversed on 2026-07-24. ConditionNames contains mistranslations,
# and lists TWO codes twice under two different names:
#   164909002 -> "left front bundle branch block" AND "left back bundle branch block"
#                (officially: left bundle branch block)
#   698252002 -> "Interior differences conduction" AND "Intraventricular block"
#                (officially: nonspecific intraventricular conduction disorder)
# A dict comprehension over that file keeps whichever occurrence comes last, so the earlier
# priority order printed a name that did not correspond to the code. Other mistranslations
# affected 17 of the statements that reach the paper's table: "ST drop down" (st depression),
# "ST extension" (st interval abnormal), "T wave opposite" (t wave inversion),
# "Sinus Irregularity" (sinus arrhythmia), "Axis left shift" (left axis deviation), etc.
CN = pd.read_csv(find("ningbo", "ConditionNames_SNOMED-CT.csv"))
SNOMED_NAME = {str(r["Snomed_CT"]): "%s (%s)" % (r["Acronym Name"], str(r["Full Name"]).strip())
               for _, r in CN.iterrows()}          # corpus names as the fallback layer
SNOMED_NAME.update(PHYSIONET_2021)                 # official SNOMED preferred terms win


def read_header(ecg_id, source):
    """Return (dx_codes, age, sex) from the record .hea header."""
    try:
        p = resolve_path(ecg_id, source) + ".hea"
    except Exception:
        return [], None, None
    if not os.path.exists(p):
        return [], None, None
    dx, age, sex = [], None, None
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.startswith("#Dx"):
            dx = [c.strip() for c in re.split(r"[,\s]+", line.split(":", 1)[1].strip()) if c.strip()]
        elif line.startswith("#Age"):
            v = line.split(":", 1)[1].strip()
            try: age = float(v)
            except Exception: age = None
        elif line.startswith("#Sex"):
            sex = line.split(":", 1)[1].strip()
    return dx, age, sex


need_ids = set(CSN.ecg_id.astype(str))
cache = None
if os.path.exists(CACHE):
    cache = pd.read_csv(CACHE, dtype={"ecg_id": str})
    if set(cache.ecg_id) >= need_ids:
        print("  reusing header scan cache (%d records)" % len(cache))
    else:
        print("  cache incomplete, rescanning"); cache = None

if cache is None:
    recs, n = [], len(CSN)
    for i, (eid, src) in enumerate(zip(CSN.ecg_id.astype(str), CSN.source), 1):
        dx, age, sex = read_header(eid, src)
        recs.append(dict(ecg_id=eid, dx="|".join(dx),
                         age=("" if age is None else age), sex=("" if sex is None else sex)))
        if i % 5000 == 0:
            print("    scanned %d/%d headers" % (i, n))
    cache = pd.DataFrame(recs)
    cache.to_csv(CACHE, index=False)
    print("  wrote header scan cache -> %s" % CACHE)

cache = cache.set_index("ecg_id")
csn_rows, missing_csn = [], 0
for eid, lab in zip(CSN.ecg_id.astype(str), CSN.label):
    if eid not in cache.index:
        missing_csn += 1; continue
    r = cache.loc[eid]
    dx = [c for c in str(r["dx"]).split("|") if c and c != "nan"]
    if not dx:
        missing_csn += 1
    comorb = [c for c in dx if c not in CSN_PREEXC]
    age = r["age"]
    csn_rows.append(dict(ecg_id=eid, label=int(lab),
                         age=(float(age) if str(age) not in ("", "nan") else None),
                         sex=(None if str(r["sex"]) in ("", "nan") else str(r["sex"]).upper()),
                         codes=dx, n_comorb=len(comorb)))
if missing_csn:
    print("  WARNING: %d CSN records with no parsed #Dx" % missing_csn)


# ---------------------------------------------------------------- summaries
def split(rows):
    return [r for r in rows if r["label"] == 1], [r for r in rows if r["label"] == 0]


def agestats(rows, censor=None):
    a = np.array([r["age"] for r in rows if r["age"] is not None], dtype=float)
    n_cens = int((a >= censor).sum()) if censor else 0
    if censor: a = a[a < censor]
    if len(a) == 0:
        return None, n_cens, int(sum(1 for r in rows if r["age"] is None))
    return (dict(n=int(len(a)), median=float(np.median(a)),
                 q1=float(np.percentile(a, 25)), q3=float(np.percentile(a, 75)),
                 min=float(a.min()), max=float(a.max())),
            n_cens, int(sum(1 for r in rows if r["age"] is None)))


def sexstats(rows, mapper):
    c = {}
    for r in rows:
        c[mapper(r["sex"])] = c.get(mapper(r["sex"]), 0) + 1
    return {k: dict(n=v, pct=round(100.0 * v / len(rows), 1)) for k, v in sorted(c.items())}


def burden(rows, exclude):
    """Distribution of the number of statements per record, ignoring `exclude`."""
    n = len(rows); b = {"0": 0, "1": 0, "2": 0, "3+": 0}; nc = []
    for r in rows:
        k = len([c for c in r["codes"] if c not in exclude])
        nc.append(k)
        b[str(k) if k < 3 else "3+"] += 1
    return dict(dist={k: dict(n=v, pct=round(100.0 * v / n, 1)) for k, v in b.items()},
                mean=round(float(np.mean(nc)), 2), median=float(np.median(nc)))


def counts(rows):
    c = {}
    for r in rows:
        for k in r["codes"]:
            c[k] = c.get(k, 0) + 1
    return c


def code_table(pos, neg, namemap, preexc, zero_lik=None):
    cp, cn = counts(pos), counts(neg)
    np_, nn = len(pos), len(neg)
    out = []
    for code in sorted(set(cp) | set(cn), key=lambda k: (-cp.get(k, 0), -cn.get(k, 0), k)):
        e = dict(code=code, name=namemap.get(code, "UNKNOWN"),
                 n_wpw=cp.get(code, 0), pct_wpw=round(100.0 * cp.get(code, 0) / np_, 1),
                 n_neg=cn.get(code, 0), pct_neg=round(100.0 * cn.get(code, 0) / nn, 2),
                 is_preexcitation=(code in preexc))
        if zero_lik is not None:
            e["n_zero_likelihood"] = zero_lik.get(code, 0)
        out.append(e)
    return out


ptb_pos, ptb_neg = split(ptb_rows)
csn_pos, csn_neg = split(csn_rows)

ptb_zero = {}
for r in ptb_rows:
    for k in r["zero_lik"]:
        ptb_zero[k] = ptb_zero.get(k, 0) + 1

ptb_tab = code_table(ptb_pos, ptb_neg, SCP_NAME, PTB_PREEXC, ptb_zero)
csn_tab = code_table(csn_pos, csn_neg, SNOMED_NAME, set(CSN_PREEXC))


def block(pos, neg, tab, censor, sexmap, preexc, baseline):
    a_p, cens_p, miss_p = agestats(pos, censor)
    a_n, cens_n, miss_n = agestats(neg, censor)
    ex_all = set(preexc)
    ex_path = set(preexc) | set(baseline)
    return dict(
        n_total=len(pos) + len(neg), n_wpw=len(pos), n_neg=len(neg),
        baseline_excluded_from_pathological=sorted(baseline),
        wpw=dict(age=a_p, age_censored_over_89=cens_p, age_missing=miss_p,
                 sex=sexstats(pos, sexmap),
                 burden=burden(pos, ex_all), burden_pathological=burden(pos, ex_path)),
        neg=dict(age=a_n, age_censored_over_89=cens_n, age_missing=miss_n,
                 sex=sexstats(neg, sexmap),
                 burden=burden(neg, ex_all), burden_pathological=burden(neg, ex_path)),
        n_distinct_codes=len(tab), codes=tab)


profile = dict(
    note=("Descriptive only, no statistical test. Codes are ECG statements, not clinical "
          "histories. Corpora are reported separately because SCP-ECG and SNOMED-CT are "
          "different vocabularies and are not pooled. Pre-excitation codes are flagged and "
          "excluded from the burden counts."),
    ptbxl=block(ptb_pos, ptb_neg, ptb_tab, 300.0,
                lambda s: {0: "male", 1: "female"}.get(s, "missing"),
                PTB_PREEXC, PTB_BASELINE),
    csn=block(csn_pos, csn_neg, csn_tab, None,
              lambda s: (s if s else "missing"),
              set(CSN_PREEXC), CSN_BASELINE))

json.dump(profile, open(os.path.join(MET, "wpw_comorbidity_profile.json"), "w"),
          indent=2, default=str)


# ---------------------------------------------------------------- console report
def show(tag, d):
    print("\n=== %s: %d records (%d WPW / %d non-WPW) ===" % (tag, d["n_total"], d["n_wpw"], d["n_neg"]))
    for cls in ("wpw", "neg"):
        s = d[cls]; a = s["age"]
        lbl = "WPW    " if cls == "wpw" else "non-WPW"
        if a:
            print("  %s age median %.0f [IQR %.0f-%.0f] range %.0f-%.0f | censored>89 %d | missing %d"
                  % (lbl, a["median"], a["q1"], a["q3"], a["min"], a["max"],
                     s["age_censored_over_89"], s["age_missing"]))
        print("  %s sex %s" % (lbl, {k: "%d (%.1f%%)" % (v["n"], v["pct"]) for k, v in s["sex"].items()}))
        for key, tag in (("burden", "all statements   "),
                         ("burden_pathological", "pathological only")):
            b = s[key]
            print("  %s %s /record: mean %.2f median %.0f | %s"
                  % (lbl, tag, b["mean"], b["median"],
                     {k: "%d (%.1f%%)" % (v["n"], v["pct"]) for k, v in b["dist"].items()}))
    print("  baseline rhythm/normality excluded from 'pathological only': %s"
          % ", ".join(d["baseline_excluded_from_pathological"]))
    print("  distinct codes: %d" % d["n_distinct_codes"])
    print("  %-12s %-50s %6s %7s %8s %7s" % ("code", "name", "n_WPW", "%WPW", "n_neg", "%neg"))
    for e in d["codes"]:
        star = " *" if e["is_preexcitation"] else "  "
        print("  %-12s %-50s %6d %6.1f%% %8d %6.2f%%%s"
              % (e["code"], e["name"][:50], e["n_wpw"], e["pct_wpw"], e["n_neg"], e["pct_neg"], star))


show("PTB-XL", profile["ptbxl"])
show("Chapman-Shaoxing-Ningbo", profile["csn"])
print("\n  * = pre-excitation code (describes the label, excluded from burden counts)")

# ---- integrity checks that must be reported, not assumed ----
unresolved = [e["code"] for e in csn_tab if e["name"] == "UNKNOWN"]
print("\n=== integrity checks ===")
print("  unresolved SNOMED codes remaining: %d %s"
      % (len(unresolved), unresolved if unresolved else ""))
print("  WPW code carried by WPW records: PTB-XL %d/%d, CSN %d/%d (must be 100%%)"
      % (sum(1 for r in ptb_pos if "WPW" in r["codes"]), len(ptb_pos),
         sum(1 for r in csn_pos if "74390002" in r["codes"]), len(csn_pos)))
ptb_neg_wpw = [r["ecg_id"] for r in ptb_neg if "WPW" in r["codes"]]
csn_neg_pre = {c: [r["ecg_id"] for r in csn_neg if c in r["codes"]] for c in CSN_PREEXC}
print("  NEGATIVE records nonetheless carrying a pre-excitation statement:")
print("    PTB-XL WPW statement: %d  -> %s" % (len(ptb_neg_wpw), ptb_neg_wpw))
for c, ids in csn_neg_pre.items():
    print("    CSN %s (%s): %d" % (c, CSN_PREEXC[c], len(ids)))
profile["integrity"] = dict(unresolved_snomed=unresolved,
                            ptbxl_negatives_with_wpw_statement=ptb_neg_wpw,
                            csn_negatives_with_preexcitation={c: len(v) for c, v in csn_neg_pre.items()})
json.dump(profile, open(os.path.join(MET, "wpw_comorbidity_profile.json"), "w"),
          indent=2, default=str)


# ---------------------------------------------------------------- LaTeX
# Display-only overrides, applied at print time and to nothing else. Two source vocabularies
# produce names that are correct but unreadable set in a table; the underlying code and counts
# are untouched. Keyed by the resolved name, so a source-file change cannot silently bypass them.
DISPLAY_OVERRIDE = {
    "Non-specific intraventricular conduction disturbance (block)":
        "Non-specific intraventricular conduction disturbance",   # SCP-ECG trailing gloss
    "S t changes":     "ST changes",       # SNOMED 55930002, lower-cased in the source mapping
    "St depression":   "ST depression",    # SNOMED 429622005
    "Qwave abnormal":  "Q wave abnormal",  # SNOMED 164917005
    "Low qrs voltages": "Low QRS voltages",# SNOMED 251146004
}


def latex_table(d, corpus_name, label, extra_caption=""):
    """One table per corpus, matching exactly what the paper prints."""
    keep = [e for e in d["codes"]
            if (e["pct_wpw"] >= MIN_PCT_IN_TABLE or e["pct_neg"] >= MIN_PCT_IN_TABLE)
            and not e["is_preexcitation"]
            and not (e["n_wpw"] == 0 and e["pct_neg"] < ZERO_ROW_MIN_PCT)]
    rows = []
    for e in keep:
        # Names arrive as "ABBR (full name)" for SNOMED and as prose for SCP-ECG; print prose in
        # both tables so the two read consistently.
        m = re.match(r"^\S+\s+\((.+)\)$", e["name"])
        nm = m.group(1) if m else e["name"]
        nm = (nm[0].upper() + nm[1:]) if nm else nm
        nm = DISPLAY_OVERRIDE.get(nm, nm)
        nm = nm.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
        rows.append("%s & %d & %.1f & %d & %.2f \\\\" % (nm, e["n_wpw"], e["pct_wpw"],
                                                         e["n_neg"], e["pct_neg"]))
    bp, bn = d["wpw"]["burden"], d["neg"]["burden"]
    pp, pn = d["wpw"]["burden_pathological"], d["neg"]["burden_pathological"]
    burd = ("\\multicolumn{5}{p{0.92\\linewidth}}{\\emph{Statements per record.} All statements: "
            "WPW mean %.2f (%.1f\\%% carry none), non-WPW mean %.2f (%.1f\\%% carry none). "
            "Excluding baseline rhythm and normality statements: WPW mean %.2f (%.1f\\%% carry "
            "none, %.1f\\%% carry three or more), non-WPW mean %.2f (%.1f\\%% carry none, "
            "%.1f\\%% carry three or more).} \\\\"
            % (bp["mean"], bp["dist"]["0"]["pct"], bn["mean"], bn["dist"]["0"]["pct"],
               pp["mean"], pp["dist"]["0"]["pct"], pp["dist"]["3+"]["pct"],
               pn["mean"], pn["dist"]["0"]["pct"], pn["dist"]["3+"]["pct"]))
    return """\\begin{table}[htbp]
\\centering
\\footnotesize
\\caption{%s diagnostic composition: statements carried by the %d WPW recordings against the
%s non-WPW recordings. A statement is listed if it reaches %.0f\\%% in either class; statements
absent from the WPW group are listed only if they reach %.0f\\%% among the non-WPW recordings,
since their absence is then informative. Percentages are of the class denominator.
Pre-excitation statements are omitted, since they describe the label rather than a concomitant
finding.%s Descriptive only; see Section~\\ref{sec:cohort} for the age confound and
Section~\\ref{sec:comorbid} for the inferential treatment.}
\\label{%s}
\\begin{tabular}{lrrrr}
\\toprule
 & \\multicolumn{2}{c}{WPW ($n=%d$)} & \\multicolumn{2}{c}{non-WPW ($n=%s$)} \\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}
Statement & n & \\%% & n & \\%% \\\\
\\midrule
%s
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}
""" % (corpus_name, d["n_wpw"], "{:,}".format(d["n_neg"]).replace(",", "{,}"),
       MIN_PCT_IN_TABLE, ZERO_ROW_MIN_PCT, extra_caption, label,
       d["n_wpw"], "{:,}".format(d["n_neg"]).replace(",", "{,}"),
       "\n".join(rows), burd)


CSN_NOTE = (" Statement names are the SNOMED-CT preferred terms resolved from the "
            "PhysioNet/Computing in Cardiology Challenge 2021 diagnosis mapping \\cite{georgia}, "
            "not the labels in the corpus's own condition-name file, which disagrees on 17 of "
            "the statements below (Section~\\ref{sec:cohort}).")
tex = ("%% auto-generated by wpw_comorbidity_profile.py -- do not edit by hand\n"
       + latex_table(profile["ptbxl"], "PTB-XL", "tab:comorbid-ptbxl")
       + "\n"
       + latex_table(profile["csn"], "Chapman-Shaoxing-Ningbo", "tab:comorbid-csn", CSN_NOTE))
open(os.path.join(MET, "wpw_comorbidity_table.tex"), "w", encoding="utf-8").write(tex)

print("\nWrote reports/metrics/wpw_comorbidity_profile.json")
print("Wrote reports/metrics/wpw_comorbidity_table.tex")
