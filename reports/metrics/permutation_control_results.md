# Permutation-control (label-shuffle) no-leakage sentinel

For each model the labels are shuffled and the native-fold (1-8) OOF is rebuilt with the model's own frozen XGBoost config (`n_shuffle=5`, seed=42). fold10 is never touched; the real frozen models are never refit (only throwaway shuffle models measure the null). PASS = real OOF AP exceeds the largest null AP by more than 0.05.

| model | real AP (folds 1-8) | null_mean | null_max | margin (real - null_max) | n_shuffle | verdict |
|-------|--------------------:|----------:|---------:|-------------------------:|:---------:|:-------:|
| M1 | 0.1982 | 0.0027 | 0.0031 | +0.1951 | 5 | PASS |
| M2 | 0.2995 | 0.0022 | 0.0029 | +0.2966 | 5 | PASS |
| M3 | 0.6188 | 0.0022 | 0.0033 | +0.6155 | 5 | PASS |
| M4 | 0.7184 | 0.0021 | 0.0024 | +0.7160 | 5 | PASS |
| M5v2 | 0.4290 | 0.0022 | 0.0025 | +0.4265 | 5 | PASS |

Interpretation: every null AP collapses toward the prevalence (~115/53540 = 0.00215); the real AP sits far above the null, so ranking cannot be reproduced from shuffled labels -> no detectable label leakage. best_model carries the equivalent control from its own notebook.