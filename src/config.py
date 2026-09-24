"""Central configuration: paths, seeds, device.

Everything else in the project imports from here. No module hardcodes a device
or a seed of its own.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
CV_DIR = OUT_DIR / "cv"
NB_DIR = ROOT / "notebooks"

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, CV_DIR, NB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "test_dataset.csv"

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
FEATURES = [
    "flow_rate_L_min",        # Q  [L/min]
    "concentration_mol_L",    # CA0 [mol/L]
    "inlet_temperature_K",    # T0 [K]
    "length_m",               # L  [m]
    "jacket_temperature_K",   # Tj [K]
]
TARGET = "overall_yield"      # percentage yield of B, [0, 100]

R_GAS = 8.314                 # J / (mol K)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> int:
    """Seed python, numpy and (if present) torch, including all CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def get_device():
    """Return the torch device to use everywhere. Never hardcode 'cpu'."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_report() -> dict:
    """Everything a reader of the notebook needs to reproduce our numbers."""
    info = {"seed": SEED}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["device"] = str(get_device())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
    except ImportError:
        info["torch"] = "not installed"
    return info


def version_table() -> "object":
    """Library versions, recorded in the final notebook."""
    import importlib

    import pandas as pd

    mods = [
        "numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch",
        "gpytorch", "xgboost", "lightgbm", "catboost", "optuna", "joblib",
    ]
    rows = []
    for m in mods:
        try:
            rows.append((m, getattr(importlib.import_module(m), "__version__", "?")))
        except ImportError:
            rows.append((m, "not installed"))
    return pd.DataFrame(rows, columns=["library", "version"])
