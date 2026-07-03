# Large model binaries

Git-LFS is intended for any frozen binary over 100 MB.

**Status: none.** Every frozen model binary in `models/` is well under 100 MB
(the largest is about 1.1 MB). All `.joblib`, `.npy`, and `.pt` model artifacts
are therefore committed directly to git; Git-LFS is not required at this time.

If a future artifact exceeds 100 MB, list it here with its size and configure
Git-LFS to track it (for example `git lfs track "*.joblib"`), then move the
entry out of direct git tracking.

Note: resumable A/B run checkpoints (`models/M7_run*/*_ckpt.npz`,
`M7_pretrain/pretrain_ckpt.pt`) are regeneration caches, not frozen artifacts.
They are excluded via `.gitignore` and will not be committed.
