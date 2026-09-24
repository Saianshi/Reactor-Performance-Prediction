"""Phases 6 and 7 -- robustness checks, then the submission file.

Nothing here re-opens a modelling decision. The model and feature set are
whatever cross-validation chose in Phase 4/5; this script only
  (a) fits that choice once on all 150 rows,
  (b) subjects the resulting test predictions to the robustness battery,
  (c) writes and verifies the submission csv.

The team name must be supplied explicitly -- the file is named after it and
we only get one submission.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

import features as ft
import models as md
import physics as ph
from config import (FEATURES, FIG_DIR, OUT_DIR, SEED, TARGET, TEST_CSV,
                    TRAIN_CSV, set_seed)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

PHYS_FIXED = {"n_flow": 0.0}
FIT_KW = dict(level="B", seed=SEED, maxiter=120, popsize=20, verbose=True)


def rule(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


# ==========================================================================
def fit_final(tr, y, feature_set="S2_core_physics", residual=True):
    """Fit the physics and the chosen model on ALL training rows."""
    rule("FIT THE FINAL MODEL ON ALL 150 ROWS")
    p, r, _ = ph.fit_physics(tr, y, fixed=PHYS_FIXED, **FIT_KW)
    print(f"    physics train RMSE = {r:.4f}")
    print("\n" + ph.check_pinning(p, fixed=PHYS_FIXED).to_string(index=False))
    print("\n" + p.summary().to_string(index=False))

    X = ft.build_features(tr, p)
    cols = [c for c in ft.FEATURE_SETS[feature_set] if c in X.columns]
    base = X["Y_phys_B"].to_numpy(float)
    gp = md.gp_matern()
    gp.fit(X[cols].to_numpy(float), y - base if residual else y)
    return p, gp, cols, base


def predict(p, gp, cols, te, residual=True):
    X = ft.build_features(te, p)
    mu, sd = gp.predict(X[cols].to_numpy(float), return_std=True)
    base = X["Y_phys_B"].to_numpy(float)
    pred = base + mu if residual else mu
    return np.clip(pred, 0.0, 100.0), sd, base


# ==========================================================================
def robustness(p, gp, cols, tr, y, te, pred, sd, base):
    rule("PHASE 6 -- ROBUSTNESS")

    print("\n[1] predictions inside the physical bounds?")
    raw = base + gp.predict(ft.build_features(te, p)[cols].to_numpy(float))
    print(f"    before clipping: min={raw.min():.4f}  max={raw.max():.4f}   "
          f"({(raw < 0).sum()} below 0, {(raw > 100).sum()} above 100)")
    print(f"    after  clipping: min={pred.min():.4f}  max={pred.max():.4f}")
    assert pred.min() >= 0 and pred.max() <= 100

    print("\n[2] do predictions vary, or collapse toward the mean?")
    print(f"    prediction sd = {pred.std():.4f}   training target sd = "
          f"{y.std():.4f}   ratio = {pred.std() / y.std():.3f}")
    print(f"    prediction range = [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"    deciles: " + " ".join(f"{v:.1f}" for v in
                                      np.percentile(pred, np.arange(0, 101, 10))))
    print("    (a ratio far below 1 would signal over-regularisation)")

    print("\n[3] +/-1% input perturbation -- predictions must move smoothly")
    rows = []
    for f in FEATURES:
        for s in (-0.01, +0.01):
            t2 = te.copy()
            t2[f] = t2[f] * (1 + s)
            p2, _, _ = predict(p, gp, cols, t2)
            rows.append({"feature": f, "perturb": f"{s:+.0%}",
                         "mean|dY|": np.abs(p2 - pred).mean(),
                         "max|dY|": np.abs(p2 - pred).max()})
    pert = pd.DataFrame(rows)
    print(pert.round(4).to_string(index=False))
    worst = pert["max|dY|"].max()
    print(f"\n    largest single-row response to a 1% input change: "
          f"{worst:.3f} yield-%")
    print("    (large jumps would indicate overfitting; the cliff region "
          "legitimately\n     responds more strongly than the flat regions)")

    print("\n[4] GP predictive uncertainty per test row")
    print(f"    sd: min={sd.min():.3f}  median={np.median(sd):.3f}  "
          f"max={sd.max():.3f}")
    thr = np.percentile(sd, 90)
    flag = np.where(sd > thr)[0]
    print(f"    rows above the 90th percentile of sd (flagged as "
          f"least certain): {flag.tolist()}")
    print(f"    {'row':>5}{'pred':>10}{'sd':>9}{'Y_phys':>10}")
    for i in flag:
        print(f"    {i:5d}{pred[i]:10.3f}{sd[i]:9.3f}{base[i]:10.3f}")

    print("\n[5] residual structure on the TRAINING rows (in-sample)")
    Xtr = ft.build_features(tr, p)
    res = y - np.clip(Xtr["Y_phys_B"].to_numpy(float), 0, 100)
    from scipy import stats as st
    for name in ["log_tau", "T_eff", "log_Da2", "tau_over_tau_opt",
                 "concentration_mol_L"]:
        if name in Xtr:
            r_, pv = st.pearsonr(Xtr[name], res)
            tag = "  <-- structure the physics missed" if pv < 0.01 else ""
            print(f"    corr(physics residual, {name:20s}) = {r_:+.4f}  "
                  f"p={pv:.4f}{tag}")
    return pert


# ==========================================================================
def write_submission(pred, team, te):
    rule("PHASE 7 -- SUBMISSION")
    path = OUT_DIR / f"{team}.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists -- refusing to overwrite. "
                         "Delete it first if you really mean to.")
    out = pd.DataFrame({TARGET: np.round(pred, 6)})
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"    written: {path}")

    back = pd.read_csv(path)
    assert back.shape == (50, 1), f"shape is {back.shape}, expected (50, 1)"
    assert list(back.columns) == [TARGET], f"columns are {list(back.columns)}"
    assert back[TARGET].notna().all()
    assert (back[TARGET] >= 0).all() and (back[TARGET] <= 100).all()
    assert len(back) == len(te)
    print(f"    verified: shape {back.shape}, column {list(back.columns)}, "
          "no NaN, all within [0,100]")
    print(f"\n    first five:\n{back.head().to_string()}")
    print(f"\n    last five:\n{back.tail().to_string()}")
    return path


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name (file is named "
                                                  "after it)")
    ap.add_argument("--feature-set", default="S2_core_physics")
    ap.add_argument("--no-write", action="store_true",
                    help="run everything but do not write the csv")
    a = ap.parse_args()

    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    y = tr[TARGET].to_numpy(float)

    p, gp, cols, base_tr = fit_final(tr, y, a.feature_set)
    pred, sd, base = predict(p, gp, cols, te)
    robustness(p, gp, cols, tr, y, te, pred, sd, base)

    res = pd.DataFrame({"pred": pred, "gp_sd": sd, "Y_phys_B": base})
    res.to_csv(OUT_DIR / "test_predictions_detail.csv", index=False)

    if a.no_write:
        print("\n    --no-write given: submission csv NOT written")
    else:
        write_submission(pred, a.team, te)


if __name__ == "__main__":
    main()
