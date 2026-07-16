# JOURNAL: WPW Detection Project, Complete Decision Log

> The single authoritative record of *why this and not that* for every real decision in the project.
> Each entry pairs a decision with the alternative it beat and the numbers that settled it. This record is
> the basis for the eventual methodological writeup. Metrics are **Average Precision (AP) on out-of-fold
> folds 1 to 8** ("OOF AP") unless stated; fold9 numbers are noisy validation (6 to 14 WPW) and are never
> the judge; fold10 is the sacred held-out test, contacted exactly once.

---

## PROJECT CONTEXT

**Objective.** Automatic detection of Wolff-Parkinson-White syndrome (WPW), a congenital accessory
atrioventricular pathway producing ventricular pre-excitation, from the standard 12-lead, 10-second
electrocardiogram. The diagnostic signature is the triad of a short PR interval (<120 ms), a delta wave
(slow, slurred QRS onset), and a widened QRS, with secondary T-wave changes.

**Datasets.** Two public 12-lead corpora are pooled: **PTB-XL** (Germany) and **Ningbo / Chapman-Shaoxing**
(China). Combined size **66,951 ECGs**, of which **142 are WPW**, a prevalence of **0.21 %** (about
**471:1** imbalance). PTB-XL contributes 70 WPW, Ningbo 72. The two corpora differ in acquisition hardware,
filtering, population, and electrode conventions, producing a strong **batch effect**.

**Split protocol.** A fixed 10-fold, patient-disjoint split (64,021 unique patients; no patient spans two
folds). **Folds 1 to 8 are training and model selection** (115 WPW), reported as pooled out-of-fold (OOF)
AP. **Fold 9 is validation** (13 WPW; noisy, looked at once). **Fold 10 is the sacred held-out test**
(14 WPW), never touched until a single, pre-registered, atomic evaluation of the frozen system.

**Central result.** Six representations (QRS-onset-morphology, global-statistical, wavelet, median-beat,
spatial-VCG, and a 1D-CNN) plus feature-union and committee models were built under one rigorous,
leakage-controlled protocol. Across all of them the ceiling is set by **data, not model diversity or
representation**: adding orthogonal detectors to the two strongest yields no committee gain, a feature-union
"super-model" loses to a two-member vote under nested cross-validation, and a deep network trained from raw
signal lands at the level of the wavelet model without helping the committee. This conclusion was reached
independently at least five times (M5, best_model v1, best_model v2, M7 Run 2, M7 Run 7).

**Deployed model.** A **two-member rank-vote of the Wavelet-localization detector (M3) and the Median-beat
morphology detector (M4)**, fusing OOF percentile ranks (`ens = 0.50 * pct(M4) + 0.50 * pct(M3)`), robust
to the batch effect because ranks transfer across corpora while absolute score scale does not. Frozen OOF AP
**0.7173** (CI95 [0.641, 0.796]), OOF AUC **0.9715**. On the once-touched **fold10** test set: **AP 0.595**
(CI95 [0.346, 0.854]), **AUC 0.950**; at the frozen operating threshold, recall 0.571 and precision 0.727.
Every model reaches AUC > 0.90 on held-out test; the modest AP is a mechanical consequence of 471:1
imbalance, and the deliverable is framed as a **risk score** (calibrated probability plus percentile), not a
fixed binary decision.

---

## MODEL NOMENCLATURE

Descriptive names are a **presentation layer only**. Every artifact on disk (model files, feature matrices,
config JSONs, OOF CSVs) keeps its `M1` to `M7` / `bestmodel` / `ensemble` ID. Cross-reference by ID.

| ID | Descriptive name | Representation | On-disk key | Role |
|----|------------------|----------------|-------------|------|
| **M1** | QRS-onset morphology detector | NeuroKit delineation; retains onset-morphology and delta-slope descriptors (no classical interval survives selection; formerly "Clinical-interval detector") | `models/M1_neurokit/` | detector (signal-limited) |
| **M2** | Global-statistical detector | per-lead distribution plus spectral summaries, no delineation | `models/M2_statistical/` | detector |
| **M3** | Wavelet-localization detector | SWT/WPT/DWT time-frequency at QRS onset | `models/M3_wavelet/` | **deployed** (ensemble member) |
| **M4** | Median-beat morphology detector | denoised median beat plus most-pre-excited beat shape | `models/M4_medianbeat/` | **deployed** (ensemble member) |
| **M5** | Spatial-VCG detector | vectorcardiogram (Kors plus inverse-Dower), delta-axis geometry | `models/M5_spatial/` | detector (excluded from ensemble) |
| **M6** | Marquette 12SL baseline | on-machine GE/Marquette measurements | `models/M6_marquette/` | **commercial reference, PTB-only (NOT one of our detectors)** |
| **M7** | 1D-CNN detector (ResNet) | learned representation from raw signal | `models/M7_run7/`, `models/M7_pretrain/` | reference-only (not deployed) |
| **best_model v1** | Feature-union model | XGBoost on union of gate-passed M1 to M5 features | `models/best_model/` | rejected (honest negative) |
| **best_model v2** | Feature-union plus clinical model | v1 union plus 54 clinical-interval features | `models/best_model_v2/` | rejected (honest negative) |
| **Deployed** | Wavelet-morphology fusion | M3 plus M4 rank-vote (alpha = 0.50) | `models/ensemble/` | **shipped system** |

---

## NOTEBOOK NUMBERING FOR THE PUBLIC REPOSITORY

Clean pipeline order (data prep, baseline, filter, detectors M1 to M7, feature-union, error analysis,
ensemble, fold10). The M7 run-series consolidates into a single numbered slot with a run suffix; legacy and
superseded notebooks move to `archive/`. Every current notebook is mapped.

| New | Current file | Content |
|-----|--------------|---------|
| `01_setup` | `01_setup.ipynb` | environment, signal loading |
| `02_eda` | `02_eda_datasets.ipynb` | dataset EDA, fold/WPW counts, patient-leakage assert |
| `03_baseline_m6` | `03_marquette_m6_combined.ipynb` | Marquette 12SL reference (M6, PTB-only) |
| `04_filter_fusion` | `04_filter_and_fusion_decision.ipynb` | 0.5-40 Hz filter plus fusion decision (absorbs OLD 06a to 06e probes) |
| `05a/b/c_m1` | `05a/05b/05c_m1_neurokit_*.ipynb` | M1 combined / PTB / Ningbo |
| `05_m1_appendix` | `05a_DOC_m1_enrichment_experiment.ipynb` | M1 172 to 697 enrichment (rejected; kept as documented negative) |
| `06a/b/c_m2` | `06a/06b/06c_m2_statistical_*.ipynb` | M2 combined / PTB / Ningbo |
| `07a/b/c_m3` | `07a/07b/07c_m3_wavelet_*.ipynb` | M3 combined / PTB / Ningbo |
| `08a/b/c_m4` | `08a/08b/08c_m4_medianbeat_*.ipynb` | M4 combined / PTB / Ningbo |
| `09a/b/c_m5` | `09a/09b/09c_m5_spatial_*.ipynb` | M5 combined / PTB / Ningbo |
| `10_m7_runN` | `m7_run1 ... run9.ipynb` | M7 CNN run-series (one factor per run) |
| `10_m7_probe / _gradcam / _scores` | `m7_probe`, `m7_gradcam`, `m7_fold9_fold10_scores.ipynb` | M7 probe, interpretability, fold9/10 reference scoring |
| `11_best_model` | `10_best_model.ipynb` / `10b_best_model_v2.ipynb` | feature-union v1 / v2 (rejected) |
| `12_error_analysis` | `11_error_analysis.ipynb` | FN/FP anatomy, QRS-width thesis, comorbidity, complementarity |
| `13_ensemble` | `12_ensemble_main.ipynb` | frozen M3 plus M4 rank-vote artifact (folds 1 to 9) |
| `14_fold10` | `13_fold10_final.ipynb` | single atomic held-out test plus per-model reference cards |
| `archive/` | `OLD - *.ipynb`, `OLD OLD - *.ipynb` | superseded legacy notebooks (07/08 legacy, OLD 03 to 06e, 00 env) |

*(The natural pipeline lands at 14 because M7 occupies its own slot and best_model is separate; if a strict
01 to 13 is required, `13_ensemble` and `14_fold10` merge into one `13_final_evaluation` notebook.)*

---

## DECISION INDEX

**Data and protocol:** [D01](#d01) dataset fusion · [D02](#d02) AP as judge · [D03](#d03) patient-leakage assert · [D04](#d04) 10-fold split and sacred fold10 · [D05](#d05) XGBoost-only

**Signal filter:** [D06](#d06) 0.5-40 Hz passband tradeoff · [D07](#d07) Butterworth order 4, zero-phase · [D08](#d08) filter denoises, does not harmonize

**Data loading:** [D09](#d09) canonical signal loader

**Batch effect:** [D10](#d10) combined model plus percentile display · [D11](#d11) standalone models assume structural overfit

**Methodology (cross-cutting):** [D12](#d12) feature gate plus dedup · [D13](#d13) two-gap overfit diagnosis · [D14](#d14) selection rule (1-SE vs max-OOF) · [D15](#d15) multiseed and 5 seeds · [D16](#d16) Platt calibration on native OOF · [D17](#d17) `evaluate_standard` · [D18](#d18) no presupposed ensemble · [D19](#d19) suspect-label keep-and-flag

**Compute and reproducibility:** [D20](#d20) compute discipline · [D21](#d21) determinism and correctness guards · [D22](#d22) `np.trapezoid`

**M6 baseline:** [D23](#d23) Marquette 12SL reference

**M1:** [D24](#d24) NeuroKit detector, K=35 · [D25](#d25) 172 to 697 enrichment rejected

**M2:** [D26](#d26) global-statistical, K=164, lr not K as anti-overfit lever

**M3:** [D27](#d27) union extraction · [D28](#d28) the v1 to v5 arc · [D29](#d29) final selection depth-4

**M4:** [D30](#d30) median-beat plus wavelet_env A/B · [D31](#d31) v2 enrichment rejected · [D32](#d32) reproducibility and freeze-guard

**M5:** [D33](#d33) v1 orthogonal but null gain · [D34](#d34) v2 dense, excluded

**M7:** [D35](#d35) scope reference-only · [D36](#d36) A/B rigor and gates · [D37](#d37) Run1 resolution plus control · [D38](#d38) Run2 C1 fail and sweep · [D39](#d39) Run7 final and override · [D40](#d40) external validity (LODO) · [D41](#d41) pretraining fails · [D42](#d42) Grad-CAM

**Feature-union:** [D43](#d43) best_model v1 rejected · [D44](#d44) best_model v2 rejected

**Composition:** [D45](#d45) ensemble = M3 plus M4, not more members

**Error analysis:** [D46](#d46) FN mechanism · [D47](#d47) the HR-pivot reversal

**Ensemble:** [D48](#d48) rank-vote on percentiles, alpha=0.50

**Fold10:** [D49](#d49) single atomic contact and result

**Deployment:** [D50](#d50) risk score not binary · [D51](#d51) percentile plus category display

**Finalization:** [D52](#d52) evaluation module rewrite, OOF schema, M2 re-key

**Paper revision:** [D53](#d53) selection not nested; fixed-feature learning curve · [D54](#d54) QRS-narrowing is real and delineator-dependent · [D55](#d55) leak-free learning curve (the paper's direct test) · [D56](#d56) finalization v_c, draft corrections, citation-order bibliography

---
# DATA AND PROTOCOL

## <a id="d01"></a>[D01] Pool PTB-XL and Ningbo into one combined training corpus
- **Context:** PTB-XL alone has only about 70 WPW, which had killed an earlier project version (unstable, too few positives). A second corpus was needed, but merging German 1990s Schiller-filtered PTB-XL with Chinese 2010s Ningbo risks a batch effect where the model learns *hospital* instead of *disease*.
- **Options considered:** (A) do nothing / single dataset (negative control); (B) common bandpass filter on both, then merge; (C) separate per-dataset models (fallback); (D) per-dataset or global normalization; (E) ComBat batch correction; (F) filter plus explicit cross-dataset verification.
- **Decision:** Merge (B plus F): combined corpus **66,951 ECGs / 142 WPW** (70 PTB, 72 Ningbo), with cross-dataset generalization as the acceptance test; keep (C) as an unused fallback.
- **Why:** Ningbo roughly doubles the positive count (70 to 142). Fusion is the *condition of existence* of the project, not a luxury. Cross-dataset transfer was later empirically confirmed on M1/M3 (AUC holds about 0.87 to 0.92 across corpora), so the fallback was never needed.
- **Rejected alternative(s):** (D) normalization and (E) ComBat both require dataset-level statistics that **do not transpose to inference on a single isolated patient's ECG**, unusable at deployment. (A) kept only as negative control. A viability check confirmed the batch effect is not aligned with the label: dataset-separability AUC on WPW = **0.828 +/- 0.079** vs on matched non-WPW = **0.893 +/- 0.057** (WPW *less* separable, so the pathology signal partly overrides the dataset signature; permutation p = 0.000).
- **Status:** frozen.

## <a id="d02"></a>[D02] Average Precision (AP) is the judge, not accuracy or AUC
- **Context:** At 471:1 imbalance, "always predict non-WPW" scores 99.79 % accuracy while detecting nothing; ROC-AUC is inflated because its x-axis (FPR) is dominated by the huge healthy majority.
- **Options considered:** accuracy; ROC-AUC as primary; AP (area under precision-recall) as primary.
- **Decision:** **AP is the primary judge everywhere**, AUC is secondary/diagnostic.
- **Why:** Worked example: catching 81 WPW at the cost of 5000 FP gives ROC FPR about 9 % (AUC about 0.93, "looks good") but PR precision 1.6 % (AP about 0.04, "catastrophe"), the same model read two opposite ways. AP concentrates on the rare class and ignores the easy-negative mass.
- **Rejected alternative(s):** Accuracy (rewards ignoring the rare class). AUC retained only as secondary; its contrast with AP is what later exposed the batch effect (AUC transfers, AP collapses). M6, originally judged on AUC 0.969, was retro-reported with AP 0.583.
- **Status:** frozen.

## <a id="d03"></a>[D03] Patient-leakage is a blocking assertion, not a warning
- **Context:** If one patient's ECGs land in two folds, the model recognizes the *patient*, not the disease: the only failure that could silently invalidate every downstream number.
- **Options considered:** non-blocking diagnostic print vs a hard assertion that halts the notebook.
- **Decision:** **Blocking assert** (notebook stops on any leak), generalized to the shared EDA notebook.
- **Why:** Verified clean: **64,021 unique patients, 66,951 ECGs, 0 patients across folds**; 142 WPW over 129 unique patients. WPW per fold {1:16, 2:16, 3:15, 4:11, 5:14, 6:14, 7:15, 8:14, 9:13, 10:14}, giving train(1 to 8)=115, val(9)=13, test(10)=14.
- **Rejected alternative(s):** a warning print (too easy to miss; leakage is existential).
- **Status:** frozen.

## <a id="d04"></a>[D04] Native 10-fold split; fold9 is noisy validation; fold10 is sacred, touched once
- **Context:** Model selection, calibration, and thresholding all need honest out-of-sample estimates; small folds (6 to 14 WPW) are individually unreliable.
- **Options considered:** shuffle-based `StratifiedKFold` over the native folds vs iterating strictly on native folds 1 to 8; judging on a single fold vs pooled OOF; using fold10 during development vs sealing it.
- **Decision:** All selection, calibration, and thresholding iterates on **native folds 1 to 8 only** (never re-shuffled, since shuffle can co-locate correlated ECGs and leak). Report **pooled OOF AP**. **Fold9** is looked at once as directional validation. **Fold10 is never touched** until the final atomic test.
- **Why:** The "fold9 trap": for M4 trained on Ningbo, fold9 read **0.908** while the stable pooled OOF (58 WPW) was **0.581**; one ECG moves a 7-WPW fold by about 0.17. On M6, an accidental shuffled-OOF protocol inflated F1 to 0.645 vs 0.471 on native folds (precision 0.597 to 0.354), pure calibration leakage; AUC barely moved (0.9738 to 0.9686), proving the ranking was fine but the operating point was leaking.
- **Rejected alternative(s):** shuffled OOF (leaks correlated samples, inflates precision/F1); single-fold judging (fold9's 13 WPW give CI about [0.13, 0.61]).
- **Status:** frozen (anti-shuffle rule graved project-wide).

## <a id="d05"></a>[D05] XGBoost as the sole detector algorithm
- **Context:** Choice of learning algorithm shared by all detectors; whether to keep Random Forest / Logistic Regression as controls.
- **Options considered:** XGBoost only; XGBoost plus RF/LR as controls; RF or LR as detectors.
- **Decision:** **XGBoost only** for every detector; RF/LR retained solely for auxiliary roles (Platt calibration, ensemble weighting). The justification is *architectural*, not "best AP".
- **Why:** XGBoost handles **NaN natively**, central to the extraction-failure hypothesis (a feature that cannot be computed on a given ECG is informative, and RF/LR would require median imputation that erases that signal). Fair tuning: RF tuned reached competitive OOF (AP 0.31 vs XGB 0.27) but generalized worse on fold9 (0.30 vs 0.41); LR plateaued at AP 0.16 (AUC 0.97 but fails to rank WPW to the top). Keeping one algorithm also keeps the M1 to M7 comparison clean (only the *representation* changes).
- **Rejected alternative(s):** RF (fold9 0.30 < 0.41; needs imputation); LR (AP 0.16); keeping controls was an initial position later **superseded**, because asymmetric tuning (XGB tuned, RF/LR at defaults) is a biased comparison and costs time for no scientific gain.
- **Status:** frozen (an initial "keep controls" position was revised to XGBoost-only).

---

# SIGNAL FILTER

## <a id="d06"></a>[D06] The 0.5-40 Hz passband: a denoise-versus-signal-loss tradeoff
- **Context:** The passband must remove noise while preserving the delta wave, whose slurred QRS onset carries components that extend above 40 Hz. Cutting at 40 Hz therefore trades denoising against a real, measurable loss of delta signal.
- **Options considered:** no filter; passband 0.5-40 Hz; passband 0.5-75 Hz; passband 0.5-100 Hz.
- **Decision:** **0.5-40 Hz passband**, applied identically to M1 to M5, M7, and deployment.
- **Why:** The 0.5 Hz high-pass removes baseline wander (respiration, motion, electrode sweat) that distorts amplitude and area; the 40 Hz low-pass removes mains interference (50/60 Hz) and EMG tremor. The cost is real: QRS-upslope energy loss has median 15.8 %, with several ECGs above 20 % (ptbxl/2145 V2 26.8 %, ningbo/JS07007 V2 24.9 %, ptbxl/4658 V1 23.8 %). Because the filter does not harmonize the batch effect anyway ([D08](#d08)), the band was decided directly on detection performance rather than on batch reduction: three maximally different probe models all favor 0.5-40 (M1 AP 0.783, M3 0.697, M7 AUC 0.867), and M1's most discriminative feature (`QRS_upslope`) has its effect size *maximized* at 0.5-40. The delta's slurred onset is mostly slow enough to survive the 40 Hz cut, so the denoising gain outweighs the loss.
- **Rejected alternative(s):** no filter (worse combined AP, larger cross-dataset drop); **0.5-75 Hz** and **0.5-100 Hz** (retain more high-frequency content but never beat 0.5-40 on detection, and a wider band lets more mains/EMG through for no detection gain; M1 combined at 0.5-75 is 0.753 < 0.783).
- **Status:** frozen.

## <a id="d07"></a>[D07] Butterworth order 4, applied zero-phase (`sosfiltfilt`)
- **Context:** Given the band, the filter's steepness (order) and its phase behavior directly reshape the QRS, which every morphology detector reads. This was left implicit and is made explicit here.
- **Options considered:** Butterworth order 2 vs 4 vs 8; causal single-pass filtering vs zero-phase forward-backward filtering (`sosfiltfilt`).
- **Decision:** **Butterworth order 4, applied zero-phase** via `sosfiltfilt` (second-order-sections form, fs 500), frozen in `filter_config.json` and imported identically at train and inference time.
- **Why:** Order 4 gives a steeper roll-off than order 2 (better mains and baseline rejection) without the passband ringing and numerical instability of order 8, which would distort the very QRS onset the detectors depend on. Zero-phase (forward-backward) filtering introduces **no phase shift**, so it does not time-shift or skew the QRS: this is essential for a morphology detector, because a phase-distorted delta onset would corrupt slope, area, and polarity features (the strongest signals in M3 and M4). The second-order-sections implementation is numerically stable at order 4.
- **Rejected alternative(s):** order 2 (roll-off too gentle, leaves more residual mains and baseline); order 8 (passband ringing and numerical instability distort the QRS); causal single-pass filtering (introduces a phase shift that skews QRS timing and morphology, unacceptable for a morphology pipeline).
- **Status:** frozen.

## <a id="d08"></a>[D08] The filter denoises; it does **not** harmonize the batch effect
- **Context:** An early hypothesis held that a bandpass would erase the PTB-versus-Ningbo difference (assumed to live in high frequencies).
- **Options considered:** keep the "filter harmonizes" hypothesis vs measure dataset separability per filter with a full classifier.
- **Decision:** **Reject harmonization.** The filter's role is denoising only; fusion viability rests on cross-dataset generalization, not on erasing the batch signature.
- **Why:** Dataset-separation AUC (non-WPW, 1500 per dataset): unfiltered **0.9487**; 0.5-40 **0.8987** (down 0.050); 0.5-100 0.8999; 1-40 0.9002. The batch effect is **massive (about 0.95)** and no bandpass brings it near 0.5 (best reduction only 0.05). An early post-filter band-energy measurement was discarded as circular (edge distortion inflated the baseline band to 0.684).
- **Rejected alternative(s):** the harmonization hypothesis, refuted by about 0.90 residual separability.
- **Status:** frozen.

---

# DATA LOADING

## <a id="d09"></a>[D09] One canonical signal loader, imported everywhere
- **Context:** PTB-XL and Ningbo store lead names differently, and per-notebook loading could silently misalign leads across the pipeline, corrupting every downstream feature.
- **Options considered:** ad-hoc per-notebook signal loading vs one canonical loader imported by every notebook and by deployment.
- **Decision:** One committed `src/signal_loading.py`, **imported (never regenerated) everywhere** (M1 to M5, M7, deployment).
- **Why:** PTB-XL writes `AVR/AVL/AVF` (uppercase), Ningbo writes `aVR/aVL/aVF`; the 12-lead order is identical but the case differs. The loader uppercases names and **actively reorders** to the canonical order `[I, II, III, AVR, AVL, AVF, V1 ... V6]` (an active reorder, not just a check, so V1 is always column 6, which M5's spatial transforms require), converts to **float32** (half the RAM, precision sufficient), and validates shape and sampling rate. A single source of truth guarantees train/inference consistency. An anti-pattern was corrected: an early notebook regenerated the loader from an inline string, so opening a later notebook first on a fresh clone crashed the import; the loader (and `evaluate_standard`) are now committed source files, never rebuilt by a notebook.
- **Rejected alternative(s):** per-notebook loading (a case mismatch would place `aVR` in the wrong column and silently corrupt M5's VCG; different notebooks could drift); regenerating the loader from a notebook (import fragility on a fresh clone).
- **Status:** frozen.

---

# BATCH EFFECT

## <a id="d10"></a>[D10] Batch effect is bidirectional and non-harmonizable, so combined model plus percentile display
- **Context:** With separability about 0.95 between corpora, does the detector transfer to an unseen hospital, and how should scores be presented?
- **Options considered:** deploy per-dataset models; deploy a combined model reporting absolute calibrated probability; deploy a combined model reporting a percentile/rank.
- **Decision:** **One combined model** trained on both corpora, **displaying a percentile/rank** rather than an absolute probability; recalibration is expected per-site.
- **Why:** Measured both directions on M1 standalones: **PTB to Ningbo** AP 0.030, AUC 0.876; **Ningbo to PTB** AP 0.093, AUC 0.909. The pattern is symmetric and diagnostic: **AUC holds (about 0.87 to 0.91), so ranking/detection transfers; AP collapses, so the absolute score scale and threshold do not.** Same signature on M2 (cross AP 0.209 / 0.218, AUC about 0.90), M3 (cross AP 0.358 to 0.575, AUC 0.918 to 0.920), M4 (cross AP 0.445 / 0.627, AUC 0.963 / 0.928). Because ranks transfer and scales do not, the deployed system fuses **percentile ranks** and communicates a percentile ("top 1 % most suspect"), with a documented local-recalibration procedure. This is also *why the ensemble is a rank-vote* (see [D48](#d48)).
- **Rejected alternative(s):** per-dataset models (do not see both scales, not deployable); absolute-probability display (threshold mis-calibrates across sites, as an F1-optimal PTB threshold lands wrong on Ningbo).
- **Status:** frozen.

## <a id="d11"></a>[D11] Standalone per-corpus models: assume and document structural overfit, do not over-regularize
- **Context:** The per-corpus standalone notebooks (07b/c, 08b/c, 09b/c) exist to demonstrate the batch effect and are trained on only about 57 WPW, where overfit is structural, not a tuning failure.
- **Options considered:** (1) assume and document the structural overfit, report the honest OOF and the cross-corpus transfer; (2) aggressively regularize (reg_lambda 10 to 20, subsample 0.5, max_depth 1) to shrink the train-OOF gap.
- **Decision:** **Option 1**: assume and document; report OOF as the stable number and never the tiny-fold AP; reframe the message as "ranking transfers, the absolute score and threshold do not."
- **Why:** These models are not deployed (only the combined model is), so cosmetically shrinking the gap costs signal and changes nothing in the demonstration. For example M2's PTB standalone has gap +0.62 on 6 WPW; the scientific point is the AP-collapse-with-AUC-hold pattern, not a clean gap. Standalone selection therefore uses max-OOF depth-open with no gap cap ([D14](#d14)), and reporting cites OOF (M4-Ningbo OOF 0.581) rather than the noisy fold9 (0.908).
- **Rejected alternative(s):** Option 2 brutal regularization (cosmetic on a non-deployed model, destroys the little signal there is, and would still not make a 57-WPW model externally valid).
- **Status:** frozen.

---
# METHODOLOGY (CROSS-CUTTING)

## <a id="d12"></a>[D12] The feature gate: |d|>0.3, FDR, bootstrap CI, cross-dataset coherence, then Spearman dedup
- **Context:** With hundreds to thousands of candidate features and only 115 training WPW, unfiltered selection guarantees false discoveries.
- **Options considered:** raw top-k by effect size; a multi-criterion statistical gate; adding stability selection.
- **Decision:** A feature passes only if it clears **all** of: Cohen's **|d| > 0.3**; **p_FDR < 0.05** (Benjamini-Hochberg on the full pool); **bootstrap 95 % CI of d excluding 0**; and (combined models only) **cross-dataset coherence** (same sign and |d|>0.2 in *each* corpus). Survivors are de-duplicated by **Spearman > 0.9** (keep strongest |d|), computed via ranks plus `np.corrcoef` (10 to 50 times faster, identical result). M5 v2 added **stability selection** (kept if selected in at least 60 % of resamples).
- **Why:** Concrete funnels: M1 172 to 73 (gate) to 50 (dedup); M2 1452 to 756 to 401; M3 v5 1797 to 1272 to 610. Cross-dataset coherence is what prevents keeping a feature that only separates WPW in one hospital (that is, learning the hospital).
- **Rejected alternative(s):** top-k by d alone (no false-discovery control); pandas Spearman (same result, 10 to 50 times slower).
- **Status:** frozen (project-wide template).

## <a id="d13"></a>[D13] Overfit is diagnosed by the train-OOF gap, with two distinct gaps named apart
- **Context:** A model can memorize 115 WPW with a big feature pool; AP_oof alone does not reveal it, and one word ("gap") was overloaded onto two different quantities.
- **Options considered:** judge on AP_oof only; cap `gap = AP_train - AP_oof`; keep a single gap definition vs split it.
- **Decision:** Diagnose via the **gap**, and name two: **`gap_train_oof` = AP_train - AP_oof** (selection-time overfit) and **`gap_train_fold9` = AP_train - AP_fold9** (validation-time). When AP_train saturates at 1.0 on large pools, the absolute gap cap is **abandoned** in favor of OOF-versus-fold9 corroboration plus inter-fold variance.
- **Why:** M5 v1 showed the danger: train about 1.0, OOF about 0.53, gap +0.47 = memorization; the honest value was about 0.45. Conversely M3 v4 onward has AP_train = 1.0 by construction (548 features, 115 WPW), so a fixed gap cap of 0.30 became meaningless and even discarded the best honest model (cost about 0.14 AP in M3 v2), hence the switch to variance and fold9 guardrails.
- **Rejected alternative(s):** AP_oof-only (hides memorization); a hard gap <= 0.30 cap once AP_train saturates (arbitrary, discards good models).
- **Status:** frozen (the two-gap naming was later hard-coded into `evaluate_standard`, see [D52](#d52)).

## <a id="d14"></a>[D14] Selection rule: 1-SE for the deployed model, max-OOF-depth-open for standalones, TIE_EPS tiebreak
- **Context:** Three incompatible selection rules had coexisted (max AP on fold9, which is biased; a hand-picked 0.03 tolerance, which is arbitrary; max-OOF under gap). "K=401" (the whole pool) and "K=246" both felt arbitrary.
- **Options considered:** max AP OOF (picks k_max); max AP on fold9; hand tolerance; **1-SE** (Hastie/Tibshirani: smallest K whose OOF AP is at least the lower bootstrap-CI bound of the best model); a small-epsilon parsimony tiebreak among true OOF ties.
- **Decision:** **1-SE for the deployed/combined model** (M1 05a, M2 06a); **max-OOF depth-open for mono-dataset standalones** (1-SE over-prunes large pools there); **`TIE_EPS = 0.01`** parsimony tiebreak among configs within 0.01 of max OOF (smallest K, then best fold9). Deployed models are additionally gap-capped; standalones are not.
- **Why:** For M2 06a, 1-SE and the AP-versus-k plateau **independently** land on K=164 (best depth-2 OOF 0.362 at K=401, bootstrap lower bound 0.276, smallest tied K=164), 59 % fewer features, gap +0.141 vs +0.245. But 1-SE on mono-dataset M2 over-prunes (06b K=96, AUC 0.759 < cross-AUC 0.906, breaking the batch-effect contrast), so standalones revert to max-OOF. The TIE_EPS rule broke the M3 v4 K=455-versus-548 tie (OOF 0.5796 vs 0.5797) toward K=455 (better fold9, 93 fewer features) *without* 1-SE's over-pruning.
- **Rejected alternative(s):** max AP on fold9 (selects on the reported fold, biased, was silently used in legacy M2 08a); hand tolerance 0.03 (arbitrary); 1-SE for standalones (over-prunes, breaks contrast).
- **Status:** frozen (one family of rules with a documented deployed-versus-standalone split).

## <a id="d15"></a>[D15] Multiseed as the stability measure; a 5-seed ensemble for the CNN
- **Context:** XGBoost (subsampling) and the CNN (weight init) are stochastic, so a single training run's AP mixes real performance with seed noise, especially dangerous at 115 WPW.
- **Options considered:** report a single seed vs report a multiseed mean and standard deviation; how many seeds.
- **Decision:** Report **multiseed AP mean and standard deviation** as an init-stability check on every frozen model; the CNN (M7) additionally uses a **5-seed ensemble**.
- **Why:** The spread across seeds separates a genuine improvement from init noise. This is exactly why the M7 A/B win criterion is a bootstrap CI and why a variant that only raised the seed-ensemble AP through dispersion (wd1e3) was rejected ([D38](#d38)). Five seeds is the adopted count: enough to estimate the init spread and to average the CNN's variance, cheap enough to run a full OOF (40 models) on a CPU-only machine. Reported spreads: M1 0.614 +/- 0.041, M3 0.682 +/- 0.034, M4 0.827 +/- 0.020, M7 per-seed 0.576 +/- 0.018.
- **Rejected alternative(s):** single-seed reporting (confounds performance with init luck; it would have "confirmed" the wd1e3 mirage in M7 Run 5).
- **Status:** frozen (5-seed ensemble measured for M7; the choice of five is reasoned, not tuned).

## <a id="d16"></a>[D16] Calibration: Platt scaling on native out-of-fold scores
- **Context:** XGBoost raw scores are not probabilities, and at 471:1 the calibration step must be both honest and leak-free.
- **Options considered:** sklearn `CalibratedClassifierCV(cv=5)` (internal shuffle) vs a Platt sigmoid fit on the native-fold OOF scores; Platt vs isotonic.
- **Decision:** **Platt scaling** (a one-input logistic regression) fit on the **native-fold OOF** scores.
- **Why:** `CalibratedClassifierCV` shuffles internally, which re-introduces exactly the fold leakage [D04](#d04) forbids. Fitting Platt on native OOF keeps the anti-shuffle discipline. A subtle and important result: at 471:1 the calibrated probabilities are almost flat (M1 Brier 0.053 raw to 0.00198 calibrated), and this flatness is calibration **success**, not failure. With 0.21 % prevalence a well-calibrated probability should be low almost everywhere, so there is nothing to "undo"; this is also why deployment shows a percentile rather than the absolute probability ([D51](#d51)).
- **Rejected alternative(s):** `CalibratedClassifierCV` shuffle (re-leaks correlated samples); isotonic regression (needs more positive support than 115 WPW to be stable).
- **Status:** frozen.

## <a id="d17"></a>[D17] `src/evaluation.py::evaluate_standard` as the single evaluation standard
- **Context:** Per-notebook inline metrics would make M1 to M7 incomparable.
- **Options considered:** inline metrics per notebook vs one shared standardized module.
- **Decision:** One committed, imported module. Construction is per-model; the **test is standardized**: threshold = F1-max on OOF applied to the evaluated fold; AP plus AUC with stratified bootstrap CI95 (2000 resamples, positives and negatives resampled separately); a fixed metrics dict, a 4-panel figure, and a `{name}_metrics.json`.
- **Why:** Guarantees that every "0.xxx" across the project is measured identically. Applied retroactively to M1 and M6.
- **Rejected alternative(s):** inline per-notebook metrics (incomparable, drift-prone).
- **Status:** frozen (rewritten during finalization, see [D52](#d52)).

## <a id="d18"></a>[D18] No presupposed ensemble: each model must defend itself standalone
- **Context:** Early on, M1's low AP was excused as "it still feeds the ensemble," which licenses weak models.
- **Options considered:** build models to feed a committee vs build each to be the best standalone detector, with the committee as a *final test*.
- **Decision:** Every model is built to be the **best standalone detector**; the ensemble is a conditional final test activated only if error-decorrelation analysis shows genuinely different misses.
- **Why:** The "weak-but-useful-to-ensemble" argument was declared invalid; it had excused an old M1 at AP 0.041. Making standalone quality the target is what later made the committee analysis ([D45](#d45)) an honest, non-circular test.
- **Rejected alternative(s):** designing for the committee (produces zombie models tuned to a tiny idiosyncratic FN set that do not generalize).
- **Status:** frozen (philosophy).

## <a id="d19"></a>[D19] Suspect labels are kept and flagged, never removed without a cardiologist
- **Context:** A few WPW labels look wrong (for example a "WPW" with a long PR, the opposite of pre-excitation), and removing them would inflate every metric.
- **Options considered:** remove suspect WPW labels vs keep them and flag them for later clinical review.
- **Decision:** **Never remove** a suspect label without a cardiologist to adjudicate; **keep and flag** instead.
- **Why:** With only 142 positives, dropping "inconvenient" labels is a silent way to overfit the evaluation to the labels we like, and the project has no clinical adjudicator. Concretely, M6's false-negative review flagged ecg_id 11190 (PR 236 ms, abnormally long, the opposite of pre-excitation) and 17072 (very deviated axes) as probable label issues, but both were kept and only flagged. The irreducible-FN floor analysis ([D46](#d46)) is honest precisely because suspect cases stay in the denominator.
- **Rejected alternative(s):** removing suspect labels (inflates AP and recall, and is unfalsifiable without a cardiologist).
- **Status:** frozen (project-wide policy).

---

# COMPUTE AND REPRODUCIBILITY

## <a id="d20"></a>[D20] Compute discipline for a memory-bound, GPU-less machine
- **Context:** All runs are on one workstation (32 GB RAM, CPU-only) with multi-gigabyte feature matrices and multi-hour extractions.
- **Options considered:** a naive in-memory `joblib` pipeline (workers return large arrays) vs a disk-oriented, chunked, checkpointed pipeline.
- **Decision:** (1) store feature matrices as **float32**; (2) each parallel worker **writes its own chunk to disk** rather than returning large arrays; (3) cap **`n_jobs` <= 6**; (4) **checkpoint every unit** (fold, seed, variant) and resume by skipping completed units; (5) **`mmap`** large matrices for reads.
- **Why:** Returning big dicts or arrays through `joblib.Parallel` triggers `WinError 1450` (insufficient system resources) on this platform, so workers must serialize to disk. float32 halves the matrix footprint at sufficient precision. `n_jobs` beyond 6 exhausts RAM (one heavy notebook at a time). Per-unit checkpoints make a reboot or dead kernel cost at most one unit of a multi-hour OOF or seed run instead of the whole run. `mmap` avoids pulling a 6 to 7 GB matrix fully into RAM. One OOM incident traced to zombie kernels holding RAM was fixed by reloading the environment plus the mmap loader (peak about 6 GB).
- **Rejected alternative(s):** in-memory `joblib` returns (`WinError 1450`); `n_jobs` 10 (OOM); no checkpointing (a crash wastes hours); float64 matrices (double the RAM for no benefit).
- **Status:** frozen (hardware-driven engineering rules).

## <a id="d21"></a>[D21] Determinism and correctness guards
- **Context:** Several silent-failure modes surfaced during long runs.
- **Options considered:** trust the runs vs add explicit guards.
- **Decision:** Fixed **`random_state=42`** on every stochastic block; a **truncated-CSV guard** (raise if train folds 1 to 8 hold fewer than 100 WPW); **by-name, not positional, feature slicing** at evaluation; validate a just-written notebook by reading it back from **real disk** because the working mount has a sync lag.
- **Why:** Determinism lets a re-run reproduce a frozen run bit-for-bit (paired with the reload-not-refit rule, [D32](#d32)). The truncated-CSV guard catches the mount sync-lag corrupting a feature matrix: a 5.5 GB file re-read as 224 rows with 0 WPW would otherwise silently train on nothing. Positional slicing caused a real M5 bug, where evaluating `X[:, :K]` instead of the by-name dedup list trained on the wrong columns and produced selection OOF 0.429 but eval 0.358; switching to by-name slicing removed the divergence. The same sync lag means a just-written `.ipynb` can read back truncated over the mount, so validation reads real disk.
- **Rejected alternative(s):** positional slicing (wrong feature set, a silent 0.07 AP gap); trusting file writes over the lagged mount (silent truncation).
- **Status:** frozen.

## <a id="d22"></a>[D22] `np.trapezoid`, not `np.trapz`
- **Context:** The extraction code integrates areas (delta and QRS area features) with numpy's trapezoidal rule.
- **Options considered:** `np.trapz` vs `np.trapezoid`.
- **Decision:** **`np.trapezoid` everywhere.**
- **Why:** `np.trapz` was removed in NumPy 2.x (the machine runs 2.0.2), so it would have crashed at the first area feature in an overnight extraction; `np.trapezoid` is the supported replacement. Caught by dry-running the extraction on a synthetic signal before the long run (the general rule: test extraction code on synthetic input before any multi-hour job).
- **Rejected alternative(s):** `np.trapz` (removed in NumPy 2.x, a latent overnight crash).
- **Status:** frozen.

---
# M6: MARQUETTE 12SL BASELINE (commercial reference, PTB-only)

## <a id="d23"></a>[D23] M6 = XGBoost on on-machine Marquette measurements, as the free reference bar
- **Context:** Modern ECG carts already emit measurements (QRS duration, PR, axes, amplitudes) via an embedded 12SL-type algorithm. If our hand-built detectors cannot beat these free numbers, they add nothing. Available on PTB-XL only.
- **Options considered:** shuffled-OOF vs native-fold protocol; depth-2 vs depth-3; operating threshold at F1-max (0.28) vs a recall >= 70 % floor (0.11); remove suspect FN labels vs keep-and-flag.
- **Decision:** M6 = XGBoost (depth 2, lr 0.05, ne 100) on 10 selected Marquette globals, **native folds**, threshold **0.28**, kept as an **external reference (not one of our detectors)** and out of the ensemble.
- **Why:** Frozen **AP 0.5834** (about 177 times base rate), **AUC 0.9686** (CI95 [0.9446, 0.9867]); at threshold 0.28 recall 0.667, precision 0.691, F1 0.679. Beats the reported cardiologist AUC (0.749) and resident (0.558). Native-fold depth-2 beats depth-3 on AUC (0.9686 vs 0.9530, the title metric).
- **Rejected alternative(s):** shuffled OOF (F1 0.645 was leakage-inflated, honest value 0.471); depth-3 (AUC 0.9530); threshold 0.11 (recall 0.702 but precision 0.354, about 2 false alarms per hit, clinically unusable, costing 0.208 F1 for 0.035 recall); removing suspect labels (kept and flagged ecg_id 11190 [PR 236 ms] and 17072 instead, per [D19](#d19)). 17 of 19 FN have proba < 0.10, so M6 is blind to minimal/latent pre-excitation rather than near-missing it.
- **Status:** frozen. *(JSON note: `m6_final_config.json` stores `average_precision` 0.5834, not an `OOF_AP` key; no fold9 field, as M6 is OOF-on-PTB only.)*

---

# M1: CLINICAL-INTERVAL DETECTOR (NeuroKit)

## <a id="d24"></a>[D24] M1 = NeuroKit delineation with a wide 172-feature pool; clinical intervals alone are empty
- **Context:** M1's angle is fine per-beat delineation (P/Q/R/S/T onsets giving PR, QRS width, morphology). The question was whether classic clinical intervals suffice.
- **Options considered:** a narrow 13-feature clinical pool; a wide 172-feature pool (clinical plus per-lead discovery families: delta, QRS morphology, amplitudes/ratios, ST/T, frequency, variability); enriching further (see [D25](#d25)).
- **Decision:** Wide **172-feature** pool (v1, canonical), gated and de-duplicated, final **K=35** under 1-SE (depth 2, lr 0.05, ne 200). Robust delta slope anchored on the R peak, not the fragile R_onset.
- **Why:** The clinical-only pool (13 features) is **empty**: 0 pass the gate on all three datasets (QRS_ms has *opposite* signs across corpora, d_ptb -0.055 vs d_nin +0.523). Thesis: the delta wave destabilizes NeuroKit delineation, so value lives in form and delta features, not textbook intervals. M1 frozen: **OOF AP 0.198**, fold9 AP 0.665, AUC 0.961, F1 0.727. `extraction_failed` *is* a usable signal here (delineation breaks 7 times more often on WPW).
- **Rejected alternative(s):** narrow clinical pool (0 features survive); K=50 (max-OOF), since 1-SE's K=35 is both more parsimonious and better on fold9 (0.665 vs 0.617).
- **Status:** frozen. *(JSON note: `m1_combined_config.json` stores fold9 0.6649 plus a `legacy_07a` block at 0.571; the OOF 0.198 lives in the OOF CSV, not this JSON. The `evaluate_standard` default `m1_ref=0.571` is stale, flagged for a later cleanup.)*

## <a id="d25"></a>[D25] M1 feature enrichment 172 to 697: tested, then rejected (signal-limited)
- **Context:** Hypothesis that M1's ceiling was too few features. Enrichment was constrained to be **orthogonal to M2** (per-beat morphology plus distribution summaries only, no spectral or global stats that M2 already computes).
- **Options considered:** enrich to 697 features vs keep the 172-feature v1.
- **Decision:** **Reject** the enrichment; return to v1 (172); keep the enriched notebook as a *documented negative*. M1 is signal-limited.
- **Why:** Apples-to-apples: enriched (697) OOF AP **0.166**, AUC 0.948, fold9 0.322; v1 (172) OOF AP **0.216**, AUC **0.972**, fold9 0.614. On the honest OOF judge the enrichment is *worse* on both AP and AUC; gains appeared only at depth-3 / gap-0.5 (memorization). The ceiling is the fragile NeuroKit delineation on the delta wave, a *signal* limit, not a feature-count limit.
- **Rejected alternative(s):** the 697-feature pool (OOF 0.166 < 0.216, AUC 0.948 < 0.972). Recorded as a permanent rejection: M1 is signal-limited and is not to be re-enriched.
- **Status:** frozen (rejected).

---

# M2: GLOBAL-STATISTICAL DETECTOR

## <a id="d26"></a>[D26] M2 = global distribution plus spectral summaries, no delineation; K=164, and learning-rate (not K) is the anti-overfit lever
- **Context:** M2's identity must be distinct from M1: summarize the whole 10 s per lead (moments, energy bands, autocorrelation-based heart rate) with **no peak or wave detection**, giving a different failure mode and an ensemble candidate.
- **Options considered:** reduce K to fight overfit vs reduce learning rate; select on AP_oof vs a gap-capped fold9; include `extraction_failed` as a feature.
- **Decision:** **K=164** (depth 2, lr 0.03, ne 300), chosen by a 2D K-by-config sweep as best fold9 AP under gap <= 0.30 (later harmonized to 1-SE, which independently returns K=164). No delineation; heart rate via autocorrelation only.
- **Why:** Frozen **OOF AP 0.2995** (equal to `AP_oof_folds1_8` 0.29947), fold9 0.399, AUC 0.918. At K=164 AP_train saturates fast, but reducing K crushes signal (fold9 0.407 at K=164 to 0.09 at K=40); the real overfit lever is the **learning rate** (lr 0.03 keeps gap 0.19; lr 0.05 gives 0.27; lr 0.1 explodes). AP-versus-k plateaus from k=164 (max only 0.358 at k=361).
- **Rejected alternative(s):** small K (fold9 collapses to 0.09; a low gap achieved by learning nothing is worthless); `extraction_failed` as a feature (M2's failures are signal-loading noise, WPW 3.9 % vs non-WPW 2.2 %, *not* disease-linked, unlike M1); localizing M2 onto the beat ("M2-v2", rejected later, since it would encroach on M4 and destroy M2's defining globality).
- **Status:** frozen. *(JSON note: `m2_final_config.json` carried K=164 and `AP_oof_folds1_8` all along; only the keyed OOF CSV was missing and was added during finalization, see [D52](#d52). An earlier internal note claiming "K/OOF_AP absent" was itself an error.)*

---

# M3: WAVELET-LOCALIZATION DETECTOR

## <a id="d27"></a>[D27] M3 = time-frequency wavelet representation via one union extraction, not a 2x2x3 tournament
- **Context:** M3 must be orthogonal to M1 (delineation) and M2 (global spectral): capture the delta as a transient localized at QRS onset, spread across scales. NeuroKit is forbidden (M1-only).
- **Options considered:** a 2x2x3 tournament (transform by mother by band, 12 cells) with cherry-pick risk; vs one **union extraction** (SWT shift-invariant core plus WPT fine mid-band plus light DWT, mothers db4 plus sym4 plus coif3) followed by the standard gate.
- **Decision:** **Union extraction** into one feature matrix, then gate, dedup, select. PyWavelets, zero NeuroKit. Feature families: energy/entropy, localization/transient, inter-scale coherence.
- **Why:** A tournament invites multiple-comparison optimism (pick the winning cell post hoc); a single pool with the shared statistical gate is honest and lets an explicit i/ii/iii ablation speak. The gate spontaneously discards the high-frequency detail bands (cD1/cD2) where the batch effect lives, a validation of the methodology.
- **Rejected alternative(s):** the 2x2x3 tournament (cherry-pick risk).
- **Status:** frozen (representation carried through all versions).

## <a id="d28"></a>[D28] The M3 v1 to v5 iteration arc: energy is dead, localization then onset-plus-polarity win
- **Context:** M3 was pushed through five versions to maximize honest OOF, each a hypothesis about *what* to extract, stopping before overfitting its own OOF.
- **Options considered (per version):** v1 coarse pool; v2 enriched (about 4700 features) plus fine k-grid; v3 drop the energy family; v4 add a QRS-mask localization family plus open depth beyond 2; v5 add onset-weighting plus delta polarity.
- **Decision:** Ship **v5** as final. Progression of fold9 AP: v1 0.260, v2 0.326, v3 0.314, v4 0.546, **v5 fold9 0.654 / OOF 0.619** (2.4 times v1).
- **Why:** The ablations are the story. **Energy is dead weight** (i_energy 0.083 vs ii_localization 0.407, localization about 4 times energy), so v3 dropped it. The **QRS-mask** family lifted loc-only 0.378 to 0.398. The decisive family is **onset-weighting plus delta polarity** (+0.092): the single strongest feature in the whole project is `swtdb4_D6_I_qrspolsigned`, **d = -2.155**, with a physiologically correct sign flip across leads (I/aVL negative, III d=+1.56, aVR d=+1.20) that replicates cross-dataset (d_ptb -2.20, d_nin -1.69). prec@recall-0.8 climbed 0.038 to 0.087 to 0.193 across v3 to v5.
- **Rejected alternative(s):** the energy family (0.083, dropped); inter-beat variability (reserved for M4 to avoid overlap); continuing to a v6 (declared past the point of overfitting M3's own OOF, since fold10 is the arbiter).
- **Status:** frozen (v5 locked and renamed to canonical `m3_features.csv` / `m3_combined_*`).

## <a id="d29"></a>[D29] M3 final selection: remove 1-SE, open depth to 4, max-OOF with TIE_EPS parsimony
- **Context:** In v3 the 1-SE rule wiped a real gain (bootstrap CI of the 0.470 model dropped to 0.375, so everything "tied" and it took K=220 / OOF 0.376). The explicit priority was to **maximize performance**.
- **Options considered:** keep 1-SE vs pure max-OOF; fix depth-2 vs open depth 2/3/4; fixed gap-cap vs variance and fold9 guardrails.
- **Decision:** **Remove 1-SE** (use max-OOF), **open depth 2/3/4**, guardrails = inter-fold variance plus fold9 corroboration (not a hard gap cap). Final **K=500, depth 4, lr 0.1**; a TIE_EPS tiebreak had earlier favored K=455 over 548 between true OOF ties.
- **Why:** Final frozen M3: **OOF AP 0.6188**, fold9 0.6537, AUC 0.929, multiseed 0.682 +/- 0.034, per-fold OOF 0.622 +/- 0.078 (range 0.515 to 0.758, no collapse). Depth-4 is stable across folds and fold9 corroborates; AP_train = 1.0 by construction, so the train-gap is correctly ignored per [D13](#d13). 1-SE had cost 0.094 AP in v3.
- **Rejected alternative(s):** 1-SE (over-prunes with wide 115-WPW bootstrap CIs); fixed depth-2 (OOF about 0.49, leaves performance on the table); K=548 = k_max (K=455/500 more parsimonious between ties). Caveat declared: K=500 plus depth-4 sits near a grid corner (selection optimism), so honest external estimate about 0.55 to 0.62, confirmed later by fold10 (0.544).
- **Status:** frozen.

---

# M4: MEDIAN-BEAT MORPHOLOGY DETECTOR (best standalone)

## <a id="d30"></a>[D30] M4 = denoised median-beat morphology on a composite R-detection; wavelet_env wins the detector A/B
- **Context:** M4's angle is the *shape* of a noise-averaged beat. It needs R-peak detection without NeuroKit (M1-only) and without inheriting M3's wavelet bias, while keeping a clean M1/M4/M5 boundary.
- **Options considered:** per-lead R (breaks M5's inter-lead coherence) vs a composite RMS-of-12-leads R vs a single lead-II reference (single point of failure); R-detector `wavelet_env` vs `pan_tompkins`; which beat views to keep.
- **Decision:** **Composite multi-lead R** on a shared timeline; R-detector **`wavelet_env`** (promoted by A/B); three beat views: median (permanent WPW), most-pre-excited beat (intermittent WPW), inter-beat variability. Calibration-invariant morphology (divided by R amplitude). Final **K=220, depth 3, lr 0.1**.
- **Why:** M4 is the **best standalone detector: OOF AP 0.718**, fold9 0.837, AUC 0.960, prec@recall-0.8 = 0.846 (vs M3's 0.19). The A/B: `wavelet_env` beats `pan_tompkins` on all four criteria (OOF 0.718 vs 0.705, fold9 0.837 vs 0.779, fail-rate 0.16 % vs 0.4 %, and *more* orthogonal to M3 with rho 0.237 vs 0.241), refuting the fear that `wavelet_env` would inherit M3. Ablation: shape 0.515 > delta_onset 0.477 > PR 0.376; QRS 0.046, ST/T 0.062.
- **Rejected alternative(s):** per-lead R (destroys inter-lead temporal coherence needed by M5); lead-II reference (single point of failure); `pan_tompkins` (loses on all four criteria); the **inter-beat variability view** (0.005 ablation, functionally **dead**, confirmed three times across combined/PTB/Ningbo), since the intermittent-WPW signal is carried by the most-pre-excited-beat view instead.
- **Status:** frozen (`wavelet_env` = v1 = deployed). *(JSON note: `m4_combined_wavelet_env_config.json` stores only `cfg` "d3_lr10" and depth 3; explicit lr/ne are not in the JSON.)*

## <a id="d31"></a>[D31] M4 v2 enrichment rejected: it made M4 more redundant with M3
- **Context:** v1's ablation showed the variability view dead; a v2 enriched with integral, difference-view (extreme minus median), QRS notch, delta-angle, and robust RMS-of-QRS normalization was built to isolate intermittent delta.
- **Options considered:** deploy v2 (1260-feature pool) vs restore v1 (279-feature pool).
- **Decision:** **Restore v1**; v2 kept only as a documented feature-enrichment ablation.
- **Why:** Apples-to-apples (same detector, same d3_lr10), v1 wins the judge and everything that matters: OOF **0.718 vs 0.702**; per-fold 0.732 +/- 0.054 vs 0.705 +/- 0.095; multiseed 0.827 vs 0.815; dataset-confound 0.522 vs 0.554; and crucially **rho(M4,M3) 0.237 to 0.300** (v2 is *more* redundant with M3), which kills committee value. v1 wins all three ensemble pairings (M1+M4 0.676, M2+M4 0.697, M3+M4 0.736). v2's only edge is fold9 AUC 0.975 (secondary, 13 noisy WPW).
- **Rejected alternative(s):** v2 (OOF 0.702 < 0.718, rho_M3 0.300 > 0.237). The "difference view captures intermittent delta" hypothesis was refuted (features survived the gate but lowered OOF and orthogonality).
- **Status:** frozen (v1 deployed).

## <a id="d32"></a>[D32] Reproducibility lesson: the overwrite incident and the freeze-guard (no-refit) rule
- **Context:** The v1 `.ipynb` was overwritten by v2 with only data and artifacts backed up (no version control), and re-fitting XGBoost on restore drifted from the exact v1 numbers.
- **Options considered:** refit v1 from data on restore vs a freeze-guard that reloads frozen artifacts and never refits; treat an `n_jobs` change as result-identical vs as a drift source.
- **Decision:** New permanent rule: **always back up the `.ipynb` itself** before any overwrite (the notebook *is* the source). And a **freeze-guard**: if frozen artifacts exist, reload them (joblib plus OOF CSV) instead of refitting; a fit runs only when the freeze is absent (`if not FROZEN:`).
- **Why:** Refitting drifted the frozen numbers (`wavelet_env` OOF 0.718 to 0.716, confusion TP10/FN3 to TP9/FN4, AUC 0.960 to 0.941), caused by XGBoost run-to-run non-determinism (`n_jobs` 10 to 6 plus parallel histogram aggregation), *not* float precision. A frozen model must be bit-exact, so reload beats refit. A guard was added: if train folds 1 to 8 hold fewer than 100 WPW, raise (catches truncated-CSV corruption from the mount sync-lag, see [D21](#d21)).
- **Rejected alternative(s):** refit-on-restore (not result-identical for XGBoost, unacceptable drift for a "frozen" model).
- **Status:** frozen (graved as a permanent project rule; freeze-guard active on M4).

---

# M5: SPATIAL-VCG DETECTOR

## <a id="d33"></a>[D33] M5 = vectorcardiogram geometry; v1 is genuinely orthogonal but adds almost nothing to the committee
- **Context:** M1 to M4 all reduce leads independently; none model the **inter-lead direction** of activation, which is the delta axis a cardiologist uses to localize the accessory pathway.
- **Options considered:** single transform (Kors *or* Dower) vs both; M5 independent vs reusing M4's beats; iterate a v2 immediately vs run the FN-complementarity check first.
- **Decision:** M5 = VCG from **both** transforms (Kors plus inverse-Dower), reusing M4's `wavelet_env` beats on a common R anchor (no per-lead realignment, which would destroy spatial coherence), global per-ECG normalization. After v1, **run the mutual-FN check before deciding on v2** (the true test, not rho or AP).
- **Why:** v1 is legitimately the **most orthogonal detector** (rho(M5,M4) = 0.191, the lowest in the project) yet its committee gain is **null: M3+M4 0.736 to M3+M4+M5 0.743, only +0.007**. All standalone signal lives in the initial-vector and timing group; the inter-lead group failed (interlead 0.004, loop_geometry 0.016, octant 0.011). Honest standalone about 0.45 (the retained config was overfit: depth-4, train 1.000, gap +0.465, OOF 0.535 optimistic).
- **Rejected alternative(s):** single transform, independent M5, per-lead realignment (breaks spatial coherence), inter-lead statistics (near-zero ablation). **Key lesson graved: low rho (0.191) does not equal marginal gain (+0.007)**, since being different on the global score distribution is not the same as recovering the hard cases (FN) the others miss.
- **Status:** frozen (v1); motivated the v2 attempt.

## <a id="d34"></a>[D34] M5 v2 (dense VCG) still plateaus at about 0.43 and is excluded from the ensemble
- **Context:** v1 was too coarse (3D loop crushed to about 10 scalars per view). v2 densified the representation (about 1600 features: 56-point trajectory, 66 lead-pairs, area vectors, sub-windows 0-20/20-40/40-60 ms, per-beat loop variability) with stability selection.
- **Options considered:** add full M5 v2 to M3+M4; add only the pure-spatial orthogonal core (95 features); keep M3+M4 and defer to the composition analysis; raw max-OOF vs gap-capped selection.
- **Decision:** **Freeze M5 v2 but keep it out of the ensemble.** Selection gap-capped to **K=443, depth 2, OOF AP 0.429**.
- **Why:** Even densified, M5 **lowers** the committee: M3+M4 0.736 to +full-M5 0.711 (down 0.025); the orthogonal 95-feature core gives down 0.018. Nested-CV unbiased OOF is 0.392 (selection optimism about 0.037), confirming v1's 0.535 was an overfitting illusion, so the true spatial ceiling is about 0.39 to 0.43. A specificity audit found 37.2 % wide-QRS/BBB among FPs vs 20.6 % base, so M5 is partly a "wide-QRS detector". Raw max-OOF would have taken K=476 / depth-4 at 0.553 (gap +0.45, overfit). The v2 selection was also kept free of any mutual-FN awareness, so it stays a fair, non-circular input to the composition analysis.
- **Rejected alternative(s):** adding M5 (down 0.025) or its orthogonal core (down 0.018) to the committee; raw max-OOF (overfit). Both v1 (`m5_combined_*`) and v2 (`m5v2_*`) are kept on disk; v2 is the reported detector.
- **Status:** frozen (M5 closed, excluded from ensemble). *(JSON note: `m5v2_combined_config.json` OOF 0.429, cfg "d2_lr03" giving depth 2; explicit lr/ne not stored.)*

---
# M7: 1D-CNN DETECTOR (ResNet)

## <a id="d35"></a>[D35] M7 = 1D-CNN on raw signal, scoped as reference-only representation learning
- **Context:** M7 is the project's only *representation-learning* model (M1 to M6 are human feature engineering). A different paradigm may make different-nature errors and thus potential complementarity. Risk: a CNN has many parameters against 142 WPW.
- **Options considered:** strict 1D-CNN raw vs a "learned" family (1D-CNN / 2D-scalogram / pretrained encoder); tune M7 for FN-recovery or orthogonality vs maximize honest standalone only; set a performance ceiling vs none.
- **Decision:** **1D-CNN (ResNet1d, 63,521 params) on raw signal** as the principal input; pretraining as a *conditional* variant; **2D-scalogram dropped**. **No FN or orthogonality tuning, no pre-set ceiling**; orthogonality is *observed* at the end (composition analysis), never an optimization target.
- **Why:** Tuning on the project's 142 WPW or FNs would overfit a tiny idiosyncratic set and double-dip; another hospital's FNs are different cases. The deployable ceiling (vote M3+M4) was already known and twice proven by nested CV, so M7's purpose is to give a rigorous answer to the natural deep-learning question, where even a negative result is valuable.
- **Rejected alternative(s):** 2D-scalogram (scalogram plus CNN reproduces M3's wavelet front-end, the least orthogonal possible); FN-targeted tuning; a "0.6 to 0.75" ceiling (abandoned as a baseless limiting belief).
- **Status:** frozen (scope).

## <a id="d36"></a>[D36] M7 A/B rigor and pre-registered gates C1/C2
- **Context:** To avoid a "zombie" model (M5's precedent) optimized past usefulness, the A/B methodology and its stop conditions were pre-registered and externally challenged over three rounds.
- **Options considered:** win threshold = inter-seed std vs bootstrap-CI on mini-OOF AP; early-stop on internal-WPW AP vs val loss; test architecture early vs last; soft criteria vs gates with mechanical consequences.
- **Decision:** **A/B changes one factor at a time**, same seeds; **win = bootstrap CI on mini-OOF AP** (not inter-seed std); **early-stop on validation loss / fixed budget**; **order = imbalance, augmentation, regularization, architecture last**. Two pre-registered gates: **C1** (after Run 2) opens pretraining *iff* OOF >= about 0.65 **and** rho(M3,M4) < about 0.5; **C2** (composition) admits M7 to the deployed ensemble *iff* a paired test shows it recovers M3-intersect-M4 FNs beyond chance **and** M3+M4+M7 >= M3+M4. Protocol identical to M1 to M6 (OOF folds 1 to 8, fold9 validation, fold10 reserve), no fold fragmentation. A capped (<=2048) non-deep XGBoost control accompanies Run 1.
- **Why:** Inter-seed std captures only init noise, not data uncertainty at about 40 WPW. **rho<0.5 is necessary but not sufficient** (M5 had rho about 0.19 and still did not help, so passing C1 means "worth the investment", not "will help"). C2-by-AP-distribution was rejected because inter-fold CIs at 13 to 16 WPW per fold are too wide to ever bite.
- **Rejected alternative(s):** inter-seed std threshold; AP-on-12-WPW early stopping (silent variance); early architecture sweep (second-order at 142 WPW); "trio-versus-pair AP distribution" for C2 (unattainable by construction).
- **Status:** frozen (pre-registered).

## <a id="d37"></a>[D37] M7 Run 1: resolution 5000 native, and the non-deep control with a deferred verdict
- **Context:** Run 1 baseline (1-split) A/B over input length; also a sanity floor and a non-deep control answering the real question ("does representation learning add anything?").
- **Options considered:** L = 1024 / 2048 / 5000 native (auto-pick returned 1024 on CI overlap); read the deep-versus-non-deep verdict at Run 1 vs defer to Run 7.
- **Decision:** **L = 5000 native** (override the auto-pick). The non-deep XGBoost-on-flattened-raw control is capped <=2048; at Run 1 it is **cost calibration plus pipeline sanity floor only**; the deep/non-deep verdict is **read at Run 7** on the tuned config with comparable variance reduction on both sides.
- **Why:** Monotone improvement on three independent metrics: 1024 fold9 AP 0.640, per-seed 0.560 +/- 0.071, gap +0.359; 2048 0.726 / 0.625 +/- 0.064 / +0.274; **5000 0.818 / 0.729 +/- 0.034 / +0.182**. Per-seed separates 2048 to 5000 at about 3 sigma, plus the delta-timing prior (20 to 50 ms needs resolution). The control scored AP 0.008 to 0.009 (deep minus non-deep +0.63 / +0.72), so the sanity floor cleared with no pipeline bug.
- **Rejected alternative(s):** 1024 (auto-pick, overridden by per-seed plus gap plus prior); reading a verdict at Run 1 (untuned baseline; the control *losing* is **ambiguous**, not a deep-wins verdict; its AUC 0.82 flags a fixed-column amplitude/dataset confound).
- **Status:** frozen (resolution 12 x 5000).

## <a id="d38"></a>[D38] M7 Run 2: honest OOF baseline, C1 gate FAILS; Runs 3 to 5 A/B sweep
- **Context:** First truly comparable OOF number and the mechanical C1 decision, then the pre-registered A/B factors.
- **Options considered:** C1 pass (open pretraining Runs 8 to 10) vs fail (keep closed); imbalance {samp/both/focal/wbce}; augmentation {none/light/rich/strong}; regularization {base/drop50/wd1e3/both_reg}.
- **Decision:** **C1 FAILS, so pretraining stays closed** (default); continue tuning. A/B outcomes: keep **`both`** (balanced sampler plus focal), adopt **`strong`** augmentation, keep **`base`** regularization.
- **Why:** Run 2 OOF **AP 0.644** (CI95 [0.556, 0.729]), AUC 0.963, gap +0.352, so M7 is about M3 (0.619), below M4 (0.718); rank-vote M3+M4 0.718 to +M7 **0.721 (+0.004 = null)**; rho(M7,M3) 0.440. C1 fails because **0.644 < 0.65** (rho 0.440 < 0.5 passes; both required). Run 3: balanced sampling is the lever (samp 0.602, both 0.572 dominate natural; focal more stable sigma 0.022). Run 4: strong aug monotone per-seed 0.484, 0.493, 0.529, 0.576 (rich to strong +0.047, about 3 sigma, motivated override). Run 5: wd1e3 had highest *ensemble* AP 0.647 but **lower per-seed** (0.570 vs base 0.576) with double the sigma, so the gain is seed dispersion, not better models, and it was rejected.
- **Rejected alternative(s):** C1 pass (OOF 0.644 < 0.65); `samp`/`focal`/`wbce`; `none`/`light`/`rich` aug; `wd1e3`/`drop50`/`both_reg` regularization (dispersion gains or degradation). This is the **third** independent confirmation that data, not diversity, is the bottleneck.
- **Status:** C1 FAIL at Run 2 (later flipped at Run 7); config `both+strong+base` frozen.

## <a id="d39"></a>[D39] M7 Run 7: final OOF 0.651; C1 re-check PASSES (fragile); a journaled override opens bounded pretraining
- **Context:** Final from-scratch OOF with the frozen config; C1 re-evaluated on the real final number.
- **Options considered:** freeze M7 now vs open bounded pretraining Runs 8 to 10.
- **Decision:** M7-v1 final = **OOF 0.651**; C1 re-check **PASSES** (0.651 >= 0.65, rho 0.454 < 0.5), so open Runs 8 to 10 **bounded**, journaled under the override rule, with a Run 9 sub-gate safeguard.
- **Why:** OOF AP **0.651** (CI95 [0.566, 0.732]), AUC 0.965, gap +0.344; delta vs Run 2 = +0.007 (the mini-OOF +0.047 of strong-aug **evaporates over 7 folds**, a data-limited plateau); fold9 standardized 0.834; rho(M7,M3) 0.454, rho(M7,M4) 0.370; rank-vote +M7 = **+0.004, invariant**. The override is justified as *C1 was honored* plus a rigorous test of transfer learning, explicitly **not** to win the ensemble.
- **Rejected alternative(s):** freezing without testing pretraining (would leave the natural transfer-learning question unanswered); interpreting the low rho as "will help" ([D36](#d36)). The PASS is flagged fragile (0.651 vs 0.65, CI centered on 0.65, a +0.007 wiggle vs Run 2's 0.644).
- **Status:** frozen (M7-v1 = 0.651). *(No `models/` JSON stores M7's OOF/fold9; they live in the M7 result logs.)*

## <a id="d40"></a>[D40] M7 external validity via leave-one-dataset-out plus per-source OOF, not standalone CNNs
- **Context:** How to assess M7's cross-corpus validity when a single-corpus CNN would train on only about 57 WPW.
- **Options considered:** build PTB-only and Ningbo-only standalone CNNs (like the feature models' b/c notebooks) vs leave-one-dataset-out plus a per-source split of the combined OOF.
- **Decision:** **LODO plus per-source OOF AP** (ptbxl vs ningbo); no standalone b/c CNN.
- **Why:** A CNN on about 57 WPW is pure noise, so a per-corpus standalone would be uninterpretable. The per-source split of the combined OOF (ptbxl 0.739 vs ningbo 0.553 at Run 7) already exposes the transfer gap on the same footing as the feature detectors, without that noise. It also matches the batch-effect story ([D10](#d10)): the Ningbo side is harder.
- **Rejected alternative(s):** standalone b/c CNNs (about 57 WPW is noise, uninterpretable).
- **Status:** frozen.

## <a id="d41"></a>[D41] M7 self-supervised pretraining (Runs 8 to 9): sub-gate FAILS, M7-v1 stays final
- **Context:** Build a reusable encoder by masked reconstruction (no WPW labels, folds 1 to 8 of both corpora), then test transfer. Anti-zombie sub-gate: fine-tune must beat from-scratch by at least about 3 sigma.
- **Options considered:** pretraining data = self-supervised on both corpora (folds 1 to 8) vs supervised PTB-only labels; and transfer = from-scratch vs frozen-encoder-plus-head vs full fine-tune.
- **Decision:** Pretrain **self-supervised on both corpora**; then the transfer **sub-gate FAILS, so STOP** (no Run 10). M7-v1 (0.651) remains final; M7 is **reference-only**, out of the deployed ensemble.
- **Why:** scratch per-seed 0.576 +/- 0.018; **frozen 0.067** (AP 0.063 = catastrophe, since reconstruction features encode general morphology with near-zero WPW/delta signal); **finetune 0.590** (+0.014, about 1 sigma, needs 3 sigma). Fine-tune about equal to scratch means the pretrained init is erased and the lock is the about 100 fine-tuning WPW, not the representation, so more epochs or another pretext would not change it. (Pretrain encoder: 20 epochs, best val reconstruction 0.5929, plateau about epoch 6.)
- **Rejected alternative(s):** supervised PTB-only pretraining (a PTB-specific representation would aggravate the cross-dataset batch asymmetry); frozen encoder (0.063 catastrophe); fine-tune (+0.014 < 3 sigma). Testing and *closing* transfer learning with proof is the **strongest possible negative result**, the fourth confirmation of the data-bottleneck thesis.
- **Status:** frozen (Run 10 cancelled; M7 reference-only).

## <a id="d42"></a>[D42] M7 Grad-CAM: confirms the delta and explains the null committee contribution
- **Context:** Interpretability with pre-registered reading criteria (delta/QRS-onset confirms; T/ST means orthogonality; edges/noise means artifact).
- **Options considered:** (interpretation against pre-registered criteria, not an A/B).
- **Decision:** Verdict: M7 **reads the delta**, and does so in the **same region as M3/M4**, which explains its rho and null vote-delta.
- **Why:** Salience by window (6 WPW TP, fold9): QRS 0.589, onset/delta 0.435, baseline 0.289, ST/T 0.198, so onset-dominant and not artifact/batch. Same delta/QRS region as the morphology detectors gives rho 0.37 to 0.45 and vote +0.004: same signal, different representation, not complementary. The per-source gap (ptbxl 0.739 > ningbo 0.553) is real transfer difficulty, not artifact cheating (baseline and ST-T salience are low).
- **Rejected alternative(s):** an artifact/batch reading (refuted by low baseline and ST-T salience).
- **Status:** frozen (figures `M7_gradcam_*`).

---

# FEATURE-UNION MODELS

## <a id="d43"></a>[D43] best_model v1 (feature-union of M1 to M5) rejected: nested CV unmasks selection optimism
- **Context:** A single XGBoost on the union of all gate-passed M1 to M5 features (3685 features), the natural "use everything" model; its in-sample OOF (0.727) appeared to beat the vote.
- **Options considered:** deploy the feature-union (in-sample 0.727) vs keep the M3+M4 vote (0.736).
- **Decision:** **Keep the vote**; freeze best_model as an honest negative, not deployed.
- **Why:** **Nested CV = 0.613 +/- 0.004 vs in-sample OOF 0.727, so selection optimism +0.114.** The honest 0.613 loses to the 4-model vote (0.685) and badly to M3+M4 (0.736). Choosing among about 1657 stability-surviving features with 115 WPW overfits *the choice* itself (high variance); the vote instead weights only 4 to 5 denoised scores ("less freedom = less overfitting = better generalization"). Shadow-feature check: 0 of 50 survive, so the *selection* is clean and the optimism is in the K/config search, which only nested CV exposes. Selected K=280, depth 4 (train 1.000, gap +0.273 < cap 0.30), so the cap alone was insufficient and nested CV caught it.
- **Rejected alternative(s):** trusting in-sample 0.727 and deploying (would ship the worse generalizer); adding M1/M5 to the vote (0.685 < M3+M4 0.736, "more models = worse").
- **Status:** frozen (honest negative). fold10 later: 0.553.

## <a id="d44"></a>[D44] best_model v2 (plus 54 clinical-interval features) rejected: clinical intervals add nothing
- **Context:** v1 pool plus 54 clinical features (per-lead and composite PR, P amplitude/area/duration, PR slope, short-PR fraction; PR composite valid on 99.8 % of ECGs).
- **Options considered:** deploy v2 with clinical intervals vs keep the vote.
- **Decision:** **Keep the vote**; v2 frozen as a second honest negative.
- **Why:** Nested CV v2 = **0.610 +/- 0.011, about equal to v1's 0.613**, still far below M3+M4 (0.736). In-sample rose 0.727 to 0.740 (looks like it beats M4 0.718) but that is only more selection optimism (+0.130). SHAP mass of clinical features = 4 %; the best clinical feature ranks 19th, and M4 already carried an equivalent PR feature. With 115 WPW the model cannot learn a generalizing PR-by-QRS interaction. A yellow flag confirmed thinness: the best shadow feature ranked 10th (vs 50th in v1), so only about 9 features clearly beat noise.
- **Rejected alternative(s):** v2 (nested 0.610 < vote 0.736; clinical SHAP 4 %).
- **Status:** frozen (honest negative). *(This is the **second and third** of the at-least-five independent confirmations that the vote/data ceiling holds.)*

---

# COMPOSITION

## <a id="d45"></a>[D45] The deployed ensemble is M3 plus M4, and no more members
- **Context:** With all detectors frozen, decide the committee composition on a symmetric FN/FP complementarity analysis (OOF folds 1 to 8, each model at its own F1-max threshold), so the conclusion is non-circular ([D18](#d18)).
- **Options considered:** add M1, M2, M5v2, or M7 to M3+M4; or keep the two-member vote.
- **Decision:** **Ensemble = M3 plus M4** (two members).
- **Why:** At each model's own threshold, the M3-intersect-M4 committee gives **TP 60, FN 27, FP-common 9**. Of the 27 WPW missed by *both*: **M5v2 recovers 1** (at the cost of 56 injected FP), **M7 recovers 2** (7 FP, and vote-delta only +0.004 = below noise), **M1 and M2 recover 0**; **24 of 27 are recovered by no one**. rho(M3,M4) = 0.237 is already the most orthogonal available pair. Parsimony wins: two members suffice.
- **Rejected alternative(s):** plus M5v2 (+1 FN for 56 FP), plus M7 (+2 FN for 7 FP but +0.004 AP), plus M1/M2 (0). No mutual-FN or FP tuning was allowed, since it would contaminate the fold10 test.
- **Status:** frozen.

---

# ERROR ANALYSIS

## <a id="d46"></a>[D46] The FN mechanism: narrow-QRS minimal pre-excitation, masked by comorbidity; specificity-complementarity
- **Context:** Understand *which* WPW the deployed committee misses and *why*, on the canonical populations (TP 60, FN 27 missed-by-both, MIXED 28, FP-common 9, FP-union 49).
- **Options considered:** (analysis, not a fork) candidate theses: QRS width, comorbidity masking, complementarity mechanism, irreducible floor.
- **Decision:** The central FN thesis is **minimal pre-excitation at narrow QRS**, compounded by **comorbidity masking**; M3 and M4 are complementary in **specificity**.
- **Why:** QRS width TP vs FN: Holm **p = 0.013** (median **101 vs 70 ms**), so the missed WPW have near-normal-width QRS (minimal/latent pre-excitation). Comorbidity enrichment FN vs TP (PTB): Fisher **p = 0.0036, OR = 15.4** (FNs are 15 times more often masked by infarct/block). Complementarity mechanism: **Jaccard(FN) 0.49 vs Jaccard(FP) 0.18**, so M3 and M4 miss *similar* hard cases but produce *distinct* false positives and thus complement each other in specificity (the mechanism behind rho=0.237). Irreducible floor: of 27 FN, PTB has 6 non-validated labels, 3 comorbid, 2 isolated-physiological; Ningbo has 16 (no validation field, a stated limit); the true physiological PTB floor is about 2 of 11.
- **Rejected alternative(s):** non-discriminant hypotheses recorded as negative results: intermittence (qrs_cv), delta slurring (inverted gradient), R-amplitude, per-lead localization, batch effect (PTB about Ningbo); BBB-enrichment among FP suggestive but non-conclusive (N=4).
- **Status:** frozen.

## <a id="d47"></a>[D47] The "HR pivot" reversal: a diluted-population artifact, documented as a lesson
- **Context:** An intermediate analysis pivoted the central thesis to heart rate; it later proved to be an artifact of the FN population definition.
- **Options considered:** pivot the thesis to HR (computed on a diluted FN = 55) vs audit the population and use the canonical FN = 27.
- **Decision:** **Reverse the HR pivot.** Central thesis stays QRS-width; HR is recorded as a **non-significant** result. Lesson graved: audit the population definition *before* concluding.
- **Why:** The HR effect was computed on **FN = 55 = "missed by at least one" model**, which includes the 28 easier MIXED cases and dilutes the signal. On the **canonical FN = 27 = "missed by both"**, HR is **not significant after Holm (p = 0.22)** and the effect disappears; QRS width becomes significant (p = 0.013, [D46](#d46)), the inverse of the earlier hypothesis.
- **Rejected alternative(s):** the HR pivot on FN=55 (population artifact). Kept in the record as a methodological near-miss, not hidden.
- **Status:** frozen (reversal; negative result).

---

# ENSEMBLE

## <a id="d48"></a>[D48] Rank-vote on OOF percentiles, equal weight alpha = 0.50 by parsimony
- **Context:** Fuse M3 and M4 into the deployed artifact (folds 1 to 9; fold10 untouched).
- **Options considered:** fuse calibrated probabilities vs percentile ranks; weight alpha on M4 chosen by argmax (0.96) vs equal weight (0.50).
- **Decision:** **Rank-vote on OOF percentiles**, `ens = alpha * pct(M4) + (1 - alpha) * pct(M3)`, **alpha = 0.50 frozen**; operating threshold = F1-max on the OOF ensemble = **0.9969**.
- **Why:** Rank/percentile fusion is **robust to the batch effect**, since ranks transfer across corpora while absolute calibrated scores do not ([D10](#d10)). The alpha sweep: argmax **alpha = 0.96 (AP 0.7244)** vs **equal-weight 0.50 (AP 0.7173)**, gain **+0.0071 < half-CI 0.039**, within noise, so parsimony picks 0.50. Frozen artifact `ensemble_config.json`: members M3+M4, alpha 0.50, **OOF AP 0.7173** (CI95 [0.641, 0.796]), **OOF AUC 0.9715**, threshold 0.9969, plus `ref_scores_M3/M4.npy` (the frozen folds-1-to-8 reference distribution used to rank fold9/10). Honest framing: the equal-weight ensemble (0.717) merely *equals* M4 alone (0.718) on OOF, so the deliverable is not an AP gain but **robustness (rank-vote) plus specificity complementarity**.
- **Rejected alternative(s):** alpha = 0.96 (+0.0071 < half-CI 0.039 = noise); calibrated-probability fusion (batch effect corrupts absolute scale). *(Discrepancy note: earlier documents cite "vote 0.736" as the deployable ceiling; that was an equal-weight rank-vote measured during M4's development, and a `vote_oof_AP` of 0.685 for the 4-model vote lives inside the best_model JSONs; the **authoritative deployed number is the frozen `ensemble_config.json` OOF AP 0.7173**. Trust the JSON.)*
- **Status:** frozen.

---

# FOLD10: THE SINGLE HELD-OUT TEST

## <a id="d49"></a>[D49] One atomic contact with fold10; the frozen ensemble is reported, per-model cards are reference-only
- **Context:** The entire system (composition, weights, threshold) was frozen on folds 1 to 9. Fold10 (14 WPW) is the one unbiased estimate and must not drive any re-decision.
- **Options considered:** re-choose composition/weights/threshold on fold10 vs keep everything frozen and report per-model cards as descriptive reference (Option 3); a single run with a HARD-STOP barrier.
- **Decision:** **Option 3**: keep the frozen ensemble, report every model's fold10 card as a *descriptive reference*, no re-decision, a single atomic run behind a HARD-STOP (fold9 validated first). **fold10 is now consumed.**
- **Why:** Deployed ensemble fold10: **AP 0.595** (CI95 [0.346, 0.854]), **AUC 0.950**; at the frozen threshold TP 8 / FP 3 / FN 6 / TN 6696 (recall 0.571, precision 0.727). OOF was 0.717, so fold10 sits inside the CI (unbiased, noisy at 14 WPW). Master table (own-OOF F1-max per model; all AP CIs overlap, so **ranking not statistically separable**): M7 fold10 0.745 [0.529, 0.940] AUC 0.978 · ENSEMBLE 0.595 [0.346, 0.854] AUC 0.950 · best_model 0.553 · M4 0.552 · M3 0.544 · M5v2 0.301 · M2 0.276 · M1 0.251. M7 numerically leads (recovers 10 of 14 vs the ensemble's 8 of 14) but the overlapping CIs make no approach separable, **the fifth confirmation that data is the bottleneck.** FN (6): 4825, 17885, 19147, 19330 (missed by every model, the hardest), JS11155, JS36160 (near-miss 0.996 vs 0.9969). FP (3): 10822, JS10031, JS10032.
- **Rejected alternative(s):** re-selecting anything on fold10 (nothing was, all pre-registered and frozen); a second contact (fold10 is consumed).
- **Status:** frozen (fold10 consumed).

---

# DEPLOYMENT

## <a id="d50"></a>[D50] The deliverable is a risk score, not a binary decision
- **Context:** A fold10 AP of 0.595 could be misread as a ranking weakness.
- **Options considered:** present a binary classifier at F1-max vs present a risk score judged by AP/AUC with a user-chosen operating point.
- **Decision:** The deliverable is a **risk score** (calibrated probability plus percentile); the threshold is the user's operating choice, not a claim.
- **Why:** Every model reaches **AUC > 0.90** on held-out test (ensemble 0.950): the score reliably sorts WPW to the top of the pile, the intended screening behavior. The **modest AP (0.595) is a mechanical consequence of 471:1 imbalance** (precision is brutal at high recall), not a ranking failure. In deployment, the output is a percentile ("top X % most suspect") with local recalibration, consistent with [D10](#d10).
- **Rejected alternative(s):** the binary-at-F1-max framing (misleading given the deferred-threshold reality).
- **Status:** frozen (reporting framing).

## <a id="d51"></a>[D51] The deployed display: a 1-to-100 percentile plus a category, never the absolute probability
- **Context:** At 471:1 the absolute calibrated probability is low almost everywhere ([D16](#d16)), so showing it to a user is misleading; a raw percentile over the reference distribution is also misleading because 99.8 % of the mass sits at low scores and would crush the useful range into the top 1 %.
- **Options considered:** display the absolute calibrated probability; display a raw percentile within the frozen folds-1-to-8 reference distribution; display a hybrid 1-to-100 score anchored on the precision-recall thresholds plus a named category.
- **Decision:** Display a **hybrid 1-to-100 score anchored on the PR-curve thresholds**, plus a **named category** (low, intermediate, high, very high). The reference distribution is the frozen folds-1-to-8 scores (about 53,500 ECGs), computed once, not per patient.
- **Why:** A raw percentile at 471:1 would compress the clinically useful range into the top 1 % (99.8 % of non-WPW sit at low scores). The hybrid instead anchors the 1-to-100 scale on clinically meaningful operating points (about 50 where precision starts to rise, about 90 at F1-max, about 99 at high precision), and the named category maps those bands to plain language. This is the concrete product form of the risk-score framing ([D50](#d50)) and of the rank-transfers-scale-does-not result ([D10](#d10)): a relative, batch-robust display with a documented per-site recalibration path.
- **Rejected alternative(s):** absolute calibrated probability (near-flat at 471:1, uninterpretable to a user); raw percentile (crushes the useful range into the top 1 %).
- **Status:** frozen (display design).

---

# FINALIZATION

## <a id="d52"></a>[D52] Evaluation module rewrite, canonical OOF schema, and M2 re-key
- **Context:** During finalization, three latent inconsistencies needed fixing without touching any model or fold10.
- **Options considered:** leave `evaluate_standard`'s single overloaded "gap" and dead params vs disambiguate and purge; grave M2's OOF assuming `.npy` order vs prove the AP first; heterogeneous per-model OOF files vs one canonical schema.
- **Decision:** Rewrite `evaluate_standard`: rename `gap_train_test` to `gap_train_fold9`, add a distinct `gap_train_oof` (via a new `ap_oof` arg), purge dead defaults `m6_ref`/`m1_ref`, add helpers `write_oof_canonical` and `permutation_control` (backup kept). Adopt one canonical OOF schema `(ecg_id, source, fold, label, proba_raw, proba_cal)` across all 9 OOF files; **re-key M2 with an order proof**.
- **Why:** Synthetic test 13 of 13 PASS (permutation control: null AP collapses to prevalence 0.242 vs real 1.000). M2 was a naked `.npy` with no key and was therefore **absent from every vote**; rather than assume order, labels were attached via the folds-1-to-8 metadata order and the AP recomputed to **0.2995**, matching the frozen 0.29947 (double confirmation) before writing `m2_combined_oof.csv`. AP was unchanged before and after migration on all 9 files (anti-corruption guardrail): M1 0.198, M2 0.299, M3 0.619, M4 0.718, M5v2 0.429, M7 0.651, best 0.727. The fold10 feature-coverage scare was also resolved by exact streaming counts (M3/M4 = 6713 ECG / 14 WPW present; a file mistaken for "0 MB empty" was a feature *list*, not a matrix), so no re-extraction was needed.
- **Rejected alternative(s):** leaving the overloaded gap (silently conflates two quantities); assuming `.npy` order (unverified, risks a wrong M2 key); the duplicate `m4_combined_wavelet_env_oof.csv` is **not** migrated, so do not use it (canonical = `m4_combined_oof.csv`). Retroactive harmonization of old `*_metrics.json` deferred to a later cleanup pass.
- **Status:** frozen.

---

# PAPER REVISION

## <a id="d53"></a>[D53] Feature selection is not nested; learning curve confirms the data bottleneck directly
- **Context:** After the paper draft, two questions on the folds 1-8 development estimates needed an honest answer: is feature selection nested inside the cross-validation, and can the data-bottleneck thesis be shown directly rather than only through indirect arguments?
- **Options considered:** claim a nested selection versus audit the code and report what it actually does; argue the bottleneck only indirectly (overlapping fold10 CIs, no representation separating, the M6 gap) versus measure it with a positive-subsample learning curve.
- **Decision:** State plainly that selection is computed once on the pooled folds 1-8, not re-nested per fold, and document the resulting mild optimism in Methods 4.3. Add an M4 learning curve over 10/25/40/55/70/85/100 percent of the 115 training WPW positives (negatives held fixed, 8 seeds, OOF AP on folds 1-8) as the direct test.
- **Why:** (1) The admission gate (Cohen's d, BH-FDR, bootstrap CI, cross-dataset coherence) and the Spearman dedup are computed once on `tr = df[df.fold.between(1,8)]`, then a separate OOF fold loop trains on the fixed selected set; this is uniform across M1 to M5 (notebooks `05a`, `06a`, `07a`, `08a`, `09a`). Each held-out fold therefore contributes about one eighth of the univariate selection statistics, and selection never consults the model's OOF score, so the optimism is mild and bounded. The selected sets were frozen before the single fold10 contact, so the headline held-out result (**fold10 AP 0.595 / AUC 0.950**) is clean; only the folds 1-8 OOF numbers carry the optimism. This is now documented rather than papered over with a nesting the code does not implement. (2) Learning curve OOF AP by fraction: **0.225, 0.484, 0.543, 0.564, 0.660, 0.688, 0.718**. Monotone and still rising at 100 percent (**+0.030 from 85 percent**, beyond the 85 percent seed spread of 0.016), with the inter-seed spread contracting from 0.091 to 0.007 and no plateau. Positives were varied, not the whole dataset, because at 471:1 the negatives are abundant and are not the scarce resource.
- **Rejected alternative(s):** claiming a nested procedure the notebooks do not implement (dishonest); re-running selection inside every fold to erase the mild optimism (a large recompute for a bounded effect that never touches the fold10 headline); varying total dataset size instead of the positive count (would confound the scarce resource, since negatives are not limiting).
- **Status:** frozen. Script `learning_curve_run.py`, figure `reports/figures/learning_curve.png`, values `reports/metrics/learning_curve.json`. Sixth and most direct confirmation of the data-bottleneck thesis; the prior five are indirect.

## <a id="d54"></a>[D54] The missed-case QRS narrowing is real (device-confirmed) and the measurement is delineator-dependent
- **Context:** The paper re-ran the FN mechanism on the *deployed committee's own* population (rank-vote at threshold 0.9969: 80 TP / 25 FP / 35 FN; PTB split 43 detected / 14 missed), not the earlier missed-by-both AND-population of [D46](#d46). A manuscript draft briefly read the narrow-QRS finding as an *instrumental artifact*, after an on-machine 12SL QRS duration appeared to show no difference (94 vs 89 ms, p=0.26). That reading was wrong.
- **Options considered:** trust the delineation proxy alone; read the 94/89 12SL number as "no real narrowing, hence artifact"; or audit which 12SL column was actually read.
- **Decision:** **The narrowing is real.** The proxy shows detected 118.0 vs missed 76.5 ms (Mann-Whitney p=0.0008); the true device QRS duration (`QRS_Dur_Global`) shows detected **140.0 vs missed 103.0 ms, p<0.001**, confirming it. The 94/89 was a **wrong-column bug** in `error_analysis_committee_population.json` (block `A2`): no column of `features_marquette.csv` reproduces 94/89, and `QRS_Dur_Global` has a WPW mean of 129 ms, so 94/89 would put *detected* WPW narrower than normal sinus (physiologically impossible). The audit script `qrs_12sl_column_audit.py` settled it.
- **Why:** Three delineators on the same 57 PTB WPW: the custom proxy (lead II) 118.0 / 76.5 (p<0.001); the device 12SL 140.0 / 103.0 (p<0.001); and the open-source **NeuroKit delineation INVERTS the sign** (96.0 / 136.5, not significant, anti-correlated with the device at rho = -0.22). The proxy also returns physiologically impossible <60 ms widths on 9 of 57 cases. Generalizable point: when the pathology degrades the very instrument used to characterize the errors, a single-delineator morphological error analysis can report the wrong sign, and only an independent measurement adjudicates. Artifacts `reports/metrics/qrs_three_delineators.json`, `qrs_12sl_column_audit.json`; figure `qrs_three_delineators.png`.
- **Rejected alternative(s):** the "instrumental artifact" reading of the draft (it rested on the 94/89 wrong column, refuted by the device's true `QRS_Dur_Global` 140/103 and by physiological consistency); asserting the narrowing from one proxy alone (delineator-dependent, wrong sign under NeuroKit).
- **Status:** frozen (corrected). Extends [D46](#d46) onto the committee population with device confirmation; supersedes the draft artifact reading.

## <a id="d55"></a>[D55] The leak-free learning curve (feature re-selection per subsample) is the paper's direct data-bottleneck test
- **Context:** The learning curve of [D53](#d53) held the *selected feature set fixed* while subsampling positives, which lets a low-data point borrow a feature set chosen with information it did not have. The paper needed a curve free of that leak.
- **Options considered:** keep the fixed-feature curve of [D53](#d53); or re-run the entire feature selection (gate, dedup, k/config) from scratch on every positive subsample, stratified by corpus.
- **Decision:** Adopt the **leak-free** curve (`learning_curve_leakfree.py`): re-run selection from scratch on each subsample of the 115 training WPW (fractions 10-100%, 8 seeds, negatives fixed, stratified by corpus), reported for the two deployed detectors M3 and M4.
- **Why:** M4 rises from **0.317 at 12 positives to 0.715 at 115** and is still rising at the full set: paired 90-vs-100% difference **+0.027 (95% CI [0.019, 0.033])**, seed spread contracting 0.092 to 0. M3 is **inconclusive** on its final segment (paired -0.007, CI [-0.023, +0.010]). The two protocols are biased in opposite directions at low n (the leak-free 0.317 sits *above* the fixed-feature 0.225 at 10%, over-selecting on 12 positives), and the qualitative "still climbing, not turned over" conclusion survives both. Bounded inference to the deployed vote: its dominant member M4 is demonstrably still improving and no member's saturation caps the committee, so the system as deployed is not shown to have saturated (the paper claims only this, not that data is the sole ceiling). Artifacts `reports/metrics/learning_curve_leakfree.json`, `learning_curve_M4.json`; figure `learning_curve_leakfree_M3M4.png`.
- **Rejected alternative(s):** the fixed-feature curve of [D53](#d53) as the headline (carries the borrowed-feature leak; retained only as the opposite-bias cross-check); extrapolating to a target corpus size (a 12-115 range cannot constrain an asymptote).
- **Status:** frozen. Supersedes [D53](#d53)'s fixed-feature curve as the paper's Figure 7 and direct test.


---

*End of decision log. This log records every design decision and the rationale that settled it; see the
repository `README.md` for the project overview, results summary, and reproduction details.*




