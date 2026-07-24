# WPW detection from 12-lead ECG

Detecting Wolff-Parkinson-White syndrome (WPW), a rare congenital pre-excitation, from the standard 12-lead ECG. The thesis of this project: at very low positive counts, **data volume, not model choice, is the bottleneck**.

## Result (stated honestly, up front)

The deployed model is a **rank-vote of two detectors** (wavelet-localization + median-beat morphology). It was evaluated **exactly once** on a held-out test fold that was never touched during development, with composition, weights, and threshold all frozen beforehand. On that test fold it reaches:

- **AUC 0.95** (test set)
- **AP 0.595** (95% CI [0.35, 0.85])

at a **471:1** class imbalance (**142 WPW in 66,951 ECGs**, prevalence 0.21%).

Why AUC is the meaningful headline here: the deliverable is a **risk score for screening** (sort the suspicious ECGs to the top), and ranking quality is what transfers. At 471:1, Average Precision is mechanically compressed (precision collapses as recall rises when negatives outnumber positives 471 to 1), so a modest AP is expected and is not a ranking failure. AUC, which measures ranking, holds at 0.95. The AP is reported openly, not hidden, and framed as a consequence of the imbalance rather than a weakness of the detector.

## Why this project is methodologically strict (the point of the repository)

- **One held-out contact.** The test fold (fold10) is contacted a single time, at the very end, with model composition, weights, and threshold frozen in advance. Nothing is selected on the test set.
- **No-leakage sentinel (all five feature detectors PASS).** A label-permutation control collapses the null average precision to the prevalence for every feature-based detector: real out-of-fold AP 0.198 (M1), 0.299 (M2), 0.619 (M3), 0.718 (M4), 0.429 (M5v2), each against a null AP of only 0.002 to 0.003. Ranking cannot be reproduced from shuffled labels, so there is no detectable label leakage.
- **Honest cross-validation.** Out-of-fold evaluation iterates on the native folds only, never re-shuffled (shuffling can co-locate correlated recordings and leak).
- **Patient-level leakage checks.** The split is patient-disjoint (64,021 unique patients, no patient in two folds), enforced by a blocking assertion.
- **A reversed result, documented not buried.** An apparent heart-rate effect was traced to a population-definition artifact (a diluted false-negative set) and reversed (heart rate is not significant, p = 0.22).
- **Multiplicity correction over the declared family.** The error analysis runs twelve tests on the same small error population, and all twelve are listed with raw and Holm-adjusted p-values. **Exactly one survives**: the QRS duration measured by the acquisition device, detected 140.0 vs missed 103.0 ms, raw p = 2.2e-06, Holm-adjusted 2.7e-05. The comorbidity-masking effect is the largest in the family and does **not** survive (adjusted p = 0.073); it is reported as an unconfirmed signal. See `reports/metrics/error_analysis_holm_family_v2.json`.
- **Selection optimism measured, not assumed.** Feature selection is computed once on the development folds rather than re-nested per fold. On the two models where a fully nested re-run was affordable, that optimism is **0.114 and 0.130 average precision** (0.727 → 0.613 and 0.740 → 0.610), so out-of-fold numbers in this repository should be read as upper estimates. The single held-out contact is unaffected, since everything was frozen before it.

Rigor is the contribution. The performance number is present, bounded, and secondary to the discipline.

## Model nomenclature

Descriptive names are a presentation layer; every file on disk uses the M1 to M7 / bestmodel IDs.

| ID | Descriptive name | Representation |
|----|------------------|----------------|
| M1 | QRS-onset morphology detector | NeuroKit delineation; no classical interval survives the gate, so all 35 retained features describe QRS-onset morphology |
| M2 | Global-statistical detector | per-lead distribution and spectral summaries |
| M3 | Wavelet-localization detector | wavelet time-frequency at the QRS onset |
| M4 | Median-beat morphology detector | denoised median beat and most-pre-excited beat shape |
| M5 | Spatial-VCG detector | vectorcardiogram (Kors, inverse-Dower), delta-axis geometry |
| M6 | Marquette 12SL baseline | on-machine measurements (proprietary commercial reference) |
| M7 | 1D-CNN detector (ResNet) | learned representation from the raw signal |
| best_model | Feature-union model | XGBoost on the union of the best M1 to M5 features |
| Deployed | Wavelet-morphology fusion | M3 + M4 rank-vote (equal weight) |

## Key results (held-out test fold, fold10, 14 WPW)

| Model | fold10 AP | 95% CI | fold10 AUC |
|-------|:---------:|:------:|:----------:|
| M7 (1D-CNN) | 0.745 | [0.529, 0.940] | 0.978 |
| **Deployed fusion (M3 + M4)** | **0.595** | **[0.346, 0.854]** | **0.950** |
| best_model (feature-union) | 0.553 | [0.313, 0.812] | 0.934 |
| M4 (median-beat) | 0.552 | [0.296, 0.828] | 0.903 |
| M3 (wavelet) | 0.544 | [0.310, 0.802] | 0.957 |
| M5v2 (spatial-VCG) | 0.301 | [0.095, 0.589] | 0.913 |
| M2 (global-statistical) | 0.276 | [0.082, 0.535] | 0.897 |
| M1 (clinical-interval) | 0.251 | [0.065, 0.498] | 0.941 |

At 14 test WPW, **every AP confidence interval overlaps**, so the ranking above is **not statistically separable**. This is the central finding: no representation, including a deep network, pulls away from the others at this data volume. Every model reaches AUC above 0.90, which is the property that matters for screening.

## Repository structure

```
.
|- config.py       repo-relative paths (works wherever the repo is cloned)
|- requirements.txt frozen dependency stack
|- JOURNAL.md      the full decision log (every design decision, why it was made, and the rejected alternative)
|- src/            canonical signal loader, evaluation module, feature extractors
|- notebooks/      pipeline notebooks (data prep, each detector, error analysis, ensemble, held-out test);
|                   m7_exploration/ holds the CNN run series
|- models/         frozen model artifacts: configs, .joblib, out-of-fold scores, fold9/fold10 scores, ensemble reference
|- data/           processed light artifacts (metadata, filter config, OOF scores, selected-feature lists) + data/README.md
|- reports/        figures/ (evaluation plots) and metrics/ (per-model *_metrics.json)
```

Raw ECG signals are **not shipped** (they are downloaded from PhysioNet, see `data/README.md`). The multi-gigabyte per-ECG **feature matrices are regenerated** by the notebooks and are not committed.

## Reproducibility and how to use

- Paths are centralized in `config.py` and resolved relative to the repository, so nothing is hardcoded to a machine.
- The dependency stack is pinned in `requirements.txt`.
- Raw data acquisition (PTB-XL and Ningbo / Chapman-Shaoxing from PhysioNet) is documented in `data/README.md`; point `RAW` at your local copy via the `WPW_RAW_DATA` environment variable if you regenerate features.
- The notebooks are **documentation artifacts** (cell outputs cleared). Full re-execution requires the raw data plus multi-gigabyte feature regeneration. The **frozen models, out-of-fold scores, and metrics JSONs are included**, so the results are fully inspectable without re-running anything.

## Datasets and attribution

- **PTB-XL** (Germany), CC BY 4.0: Wagner P., Strodthoff N., Bousseljot R.-D., et al. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154 (2020).
- **Ningbo / Chapman-Shaoxing** (China), via PhysioNet: Zheng J., et al. A 12-lead electrocardiogram database for arrhythmia research (Chapman University and Shaoxing People's Hospital), and its Ningbo extension.
- **PhysioNet platform**: Goldberger A.L., Amaral L.A.N., Glass L., et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 101(23):e215-e220 (2000).
- **Marquette 12SL** is a proprietary commercial algorithm, used here only as a comparison baseline (M6), not redistributed.

## License

MIT (see `LICENSE`).
