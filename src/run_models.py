"""Phases 4 and 5 -- evaluate the model zoo under one honest protocol.

STRUCTURE. The expensive object in this project is the physics fit, not the
regressors. So the loop is inverted relative to the usual sklearn pattern:

    for each fold:
        fit the PHYSICS on the fold's training rows      (once)
        build engineered features from those parameters  (once)
        for each model:
            fit on the fold's training rows, score on the held-out rows

Fitting the physics once per fold and sharing it across models is both correct
and ~12x cheaper than refitting it inside every model. It is correct because
the physics parameters are fitted by minimising TRAINING RMSE -- they see the
target -- so they must be refitted per fold, but they do not depend on which
regressor is used afterwards.

Nothing is fitted outside the fold: not the physics, not the scaler, not the
feature selection.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

import features as ft
import models as md
import physics as ph
from config import CV_DIR, OUT_DIR, SEED, TARGET, TEST_CSV, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

# the physics variant chosen in Phase 2, set from the saved fit
PHYS_FIXED = {"n_flow": 0.0}
FIT_KW = dict(level="B", seed=SEED, maxiter=50, popsize=12, verbose=False)


def rule(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


# ==========================================================================
def model_zoo(n_feat: int) -> dict:
    """name -> (estimator, feature_set, residual?)

    `residual=True` means the model predicts y - Y_phys_B and the physics
    prediction is added back, i.e. the physics acts as the GP's prior mean.
    """
    return {
        "physics only (Level B)":        (None, None, False),
        "GP Matern5/2 on residual":      (md.gp_matern(), "S2_core_physics", True),
        "GP Matern5/2 resid (S3)":       (md.gp_matern(), "S3_physics_plus", True),
        "GP Matern3/2 on residual":      (md.gp_matern32(), "S2_core_physics", True),
        "GP RQ on residual":             (md.gp_rq(), "S2_core_physics", True),
        "GP Matern5/2 zero-mean":        (md.gp_matern(), "S2_core_physics", False),
        "GP zero-mean, collapsed 4":     (md.gp_matern(), "S1_collapsed", False),
        "KRR rbf on residual":           (md.krr(alpha=1e-2), "S2_core_physics", True),
        "ExtraTrees on residual":        (md.extra_trees(), "S2_core_physics", True),
        "RandomForest on residual":      (md.random_forest(), "S2_core_physics", True),
        "XGBoost on residual":           (md.xgb(), "S2_core_physics", True),
        "LightGBM on residual":          (md.lgbm(), "S2_core_physics", True),
        "CatBoost on residual":          (md.catb(), "S2_core_physics", True),
        "MLP on residual":               (md.mlp(), "S2_core_physics", True),
        "XGBoost raw features":          (md.xgb(), None, False),
    }


def evaluate(df: pd.DataFrame, y: np.ndarray, n_splits=10, n_repeats=3,
             seed=SEED, verbose=True):
    """Run the whole zoo through RepeatedKFold with per-fold physics refits."""
    names = list(model_zoo(0).keys())
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                        random_state=seed)
    oof = {n: np.zeros((n_repeats, len(y))) for n in names}
    fold_params = []

    t_start = time.time()
    for k, (tri, tei) in enumerate(rkf.split(df)):
        rep = k // n_splits
        d_tr, d_te = df.iloc[tri], df.iloc[tei]

        # ---- physics, fitted on the fold's TRAINING rows only ----------
        p, r_phys, _ = ph.fit_physics(d_tr, y[tri], fixed=PHYS_FIXED, **FIT_KW)
        fold_params.append({**{n: getattr(p, n) for n in ph.PARAM_NAMES},
                            "fold": k, "phys_train_rmse": r_phys})

        X_tr = ft.build_features(d_tr, p)
        X_te = ft.build_features(d_te, p)
        base_tr = X_tr["Y_phys_B"].to_numpy(float)
        base_te = X_te["Y_phys_B"].to_numpy(float)

        for name, (est, fset, resid) in model_zoo(0).items():
            if est is None:
                oof[name][rep, tei] = np.clip(base_te, 0, 100)
                continue
            cols = (ft.FEATURE_SETS[fset] if fset else list(X_tr.columns))
            cols = [c for c in cols if c in X_tr.columns]
            A, B = X_tr[cols].to_numpy(float), X_te[cols].to_numpy(float)
            e = clone(est)
            if resid:
                e.fit(A, y[tri] - base_tr)
                pred = base_te + e.predict(B)
            else:
                e.fit(A, y[tri])
                pred = e.predict(B)
            oof[name][rep, tei] = np.clip(pred, 0, 100)

        if verbose and (k + 1) % n_splits == 0:
            print(f"    repeat {rep + 1}/{n_repeats} done "
                  f"({time.time() - t_start:.0f}s elapsed)")

    rows = []
    for n in names:
        per = np.array([ph.rmse(oof[n][r], y) for r in range(n_repeats)])
        rows.append({"model": n, "CV_RMSE": per.mean(),
                     "sd_repeats": per.std(ddof=1) if n_repeats > 1 else 0.0,
                     "se": (per.std(ddof=1) / np.sqrt(n_repeats)
                            if n_repeats > 1 else 0.0),
                     "best_repeat": per.min(), "worst_repeat": per.max()})
    return (pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True),
            oof, pd.DataFrame(fold_params))


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)

    rule("PHASE 4/5 -- MODEL ZOO UNDER REPEATED CV WITH PER-FOLD PHYSICS REFIT")
    print(f"    150 training rows, target sd = {y.std():.3f}")
    print("    physics refitted from scratch inside every fold; scalers and")
    print("    all preprocessing likewise. No test data touched.\n")

    t, oof, folds = evaluate(tr, y, n_splits=10, n_repeats=3)

    rule("RESULTS  (mean CV RMSE over repeats, +/- sd across repeats)")
    print(t.round(4).to_string(index=False))
    t.to_csv(CV_DIR / "model_zoo.csv", index=False)
    folds.to_csv(CV_DIR / "model_zoo_fold_params.csv", index=False)

    best = t.iloc[0]
    thr = best.CV_RMSE + best.se
    rule("WITHIN ONE STANDARD ERROR OF THE BEST")
    print("    Tie-break rule: among these, prefer the simpler and more")
    print("    physically grounded model.\n")
    print(t[t.CV_RMSE <= thr].round(4).to_string(index=False))

    rule("PAIRED COMPARISON AGAINST THE PHYSICS-ONLY BASELINE")
    ybase = oof["physics only (Level B)"]
    nrep = ybase.shape[0]
    base_per = np.array([ph.rmse(ybase[r], y) for r in range(nrep)])
    print(f"    {'model':34s}{'diff vs physics':>17}{'se':>9}{'p':>9}")
    print("    " + "-" * 69)
    for n in t.model:
        if n == "physics only (Level B)":
            continue
        per = np.array([ph.rmse(oof[n][r], y) for r in range(nrep)])
        d = per - base_per
        se = d.std(ddof=1) / np.sqrt(nrep) if nrep > 1 else 0.0
        p = stats.ttest_rel(per, base_per).pvalue if nrep > 2 else np.nan
        print(f"    {n:34s}{d.mean():+17.4f}{se:9.4f}{p:9.4f}")

    np.save(OUT_DIR / "oof_zoo.npy",
            np.stack([oof[n] for n in t.model]))
    (OUT_DIR / "oof_zoo_names.json").write_text(json.dumps(list(t.model)))
    print(f"\n    saved to {CV_DIR / 'model_zoo.csv'}")


if __name__ == "__main__":
    main()
