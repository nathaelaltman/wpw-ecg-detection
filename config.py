"""
config.py -- central path configuration for the WPW-detection repository.

All paths are resolved RELATIVE TO THIS FILE's location, so the repository works
wherever it is cloned or moved: there is no hardcoded absolute path. Every module
and notebook should import its paths from here (e.g. `from config import PROCESSED`).

Raw ECG signals are NOT shipped in the repo (they are downloaded from PhysioNet;
see data/README.md). If you regenerate the feature matrices, point RAW at your
local copy of the raw data via the `WPW_RAW_DATA` environment variable:

    export WPW_RAW_DATA=/path/to/raw        # Linux / macOS
    setx    WPW_RAW_DATA  D:\path\to\raw    # Windows

Otherwise RAW defaults to `data/raw/` inside the repo.
"""
from pathlib import Path

# Repo root = the folder containing this file
ROOT = Path(__file__).resolve().parent

SRC       = ROOT / "src"
DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS    = ROOT / "models"
REPORTS   = ROOT / "reports"
FIGURES   = REPORTS / "figures"
METRICS   = REPORTS / "metrics"
NOTEBOOKS = ROOT / "notebooks"
DOCS      = ROOT / "docs"

# Raw data is NOT shipped in the repo (downloaded from PhysioNet, see data/README.md).
# Point RAW at a local path via env var if you regenerate features:
import os
RAW = Path(os.environ.get("WPW_RAW_DATA", DATA / "raw"))
