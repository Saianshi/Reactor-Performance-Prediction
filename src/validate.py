"""Phase 5 -- validation discipline.

The single rule this module exists to enforce: NOTHING that touches the target
may be fitted outside the fold. That includes the scaler, the feature
selection, AND the physics parameters. Refitting the physics inside every fold
is the expensive part, but skipping it would let 150 target values leak into
the feature construction and make every downstream number optimistic.

Protocols provided
  repeated_cv  RepeatedKFold(10 x 10), mean RMSE and sd ACROSS REPEATS
               (the sd across repeats measures protocol stability, not
                sampling error -- both are reported and clearly labelled)
  loo_cv       leave-one-out, for the top models on 150 rows
  nested_cv    outer scoring loop with an inner tuning loop, so any model
               whose hyperparameters were tuned is reported honestly
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold

import features as ft
import physics as ph
from config import SEED, TARGET

warnings.filterwarnings("ignore")


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ==========================================================================
# A pipeline that refits the PHYSICS inside the fold
# ==========================================================================
class PhysicsPipeline(BaseEstimator, RegressorMixin):
    """Refit physics -> build features -> fit a regressor. All inside a fold.

    Parameters
    ----------
    model : an sklearn-style regressor operating on the engineered matrix
    feature_set : key into features.FEATURE_SETS, or None for all columns
    refit_physics : if False, use `params` as given (much faster; used only
        where we have already shown the parameters are stable across folds)
    residual : if True the model predicts the RESIDUAL y - Y_phys and the
        physics prediction is added back. This is the GP-on-physics-residual
        design, i.e. the physics acts as the prior mean.
    """

    def __init__(self, model, params: ph.PhysParams, feature_set: str | None =
                 "S2_core_physics", refit_physics: bool = True,
                 fixed: dict | None = None, residual: bool = False,
                 residual_col: str = "Y_phys_B", level: str = "A",
                 clip: tuple = (0.0, 100.0)):
        self.model = model
        self.params = params
        self.feature_set = feature_set
        self.refit_physics = refit_physics
        self.fixed = fixed
        self.residual = residual
        self.residual_col = residual_col
        self.level = level
        self.clip = clip

    # -- internals ----------------------------------------------------
    def _build(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        X = ft.build_features(df_raw, self.params_)
        if self.feature_set is not None:
            cols = ft.FEATURE_SETS[self.feature_set]
            X = X[[c for c in cols if c in X.columns]]
        return X

    def fit(self, df_raw: pd.DataFrame, y):
        y = np.asarray(y, float)
        if self.refit_physics:
            self.params_, self.phys_train_rmse_, _ = ph.fit_physics(
                df_raw, y, level=self.level, fixed=self.fixed or {},
                seed=SEED, maxiter=250, popsize=20, verbose=False)
        else:
            self.params_ = self.params
            self.phys_train_rmse_ = np.nan

        X = self._build(df_raw)
        self.columns_ = list(X.columns)
        self.model_ = clone(self.model)
        if self.residual:
            self.base_train_ = X[self.residual_col].to_numpy(float)
            self.model_.fit(X.to_numpy(float), y - self.base_train_)
        else:
            self.model_.fit(X.to_numpy(float), y)
        return self

    def predict(self, df_raw: pd.DataFrame):
        X = self._build(df_raw)[self.columns_]
        p = self.model_.predict(X.to_numpy(float))
        if self.residual:
            p = p + X[self.residual_col].to_numpy(float)
        return np.clip(p, *self.clip)


class PhysicsOnly(BaseEstimator, RegressorMixin):
    """The physics prior used directly as a predictor -- the baseline to beat."""

    def __init__(self, level: str = "A", fixed: dict | None = None):
        self.level = level
        self.fixed = fixed

    def fit(self, df_raw: pd.DataFrame, y):
        self.params_, self.train_rmse_, _ = ph.fit_physics(
            df_raw, np.asarray(y, float), level=self.level,
            fixed=self.fixed or {}, seed=SEED, maxiter=250, popsize=20,
            verbose=False)
        return self

    def predict(self, df_raw: pd.DataFrame):
        fn = ph.level_a if self.level == "A" else (
            lambda p, d: ph.level_b(p, d, n_steps=400))
        return np.clip(fn(self.params_, df_raw), 0.0, 100.0)


# ==========================================================================
# Protocols
# ==========================================================================
@dataclass
class CVResult:
    name: str
    mean_rmse: float          # mean over repeats of the per-repeat RMSE
    sd_repeats: float         # sd ACROSS repeats -> protocol stability
    se_mean: float            # standard error of that mean
    per_repeat: np.ndarray
    oof: np.ndarray           # OOF predictions from the FIRST repeat
    fold_rmses: np.ndarray    # every individual fold RMSE

    def line(self) -> str:
        return (f"{self.name:44s} {self.mean_rmse:8.4f} +/- "
                f"{self.sd_repeats:6.4f}   (se {self.se_mean:.4f})")


def repeated_cv(name: str, est, df_raw: pd.DataFrame, y: np.ndarray,
                n_splits: int = 10, n_repeats: int = 10, seed: int = SEED,
                verbose: bool = True) -> CVResult:
    """RepeatedKFold with the estimator refitted from scratch in every fold."""
    y = np.asarray(y, float)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                        random_state=seed)
    per_repeat = np.zeros(n_repeats)
    oof_all = np.zeros((n_repeats, len(y)))
    fold_rmses = []
    for k, (tri, tei) in enumerate(rkf.split(df_raw)):
        rep = k // n_splits
        e = clone(est)
        e.fit(df_raw.iloc[tri], y[tri])
        p = e.predict(df_raw.iloc[tei])
        oof_all[rep, tei] = p
        fold_rmses.append(rmse(p, y[tei]))
    for r in range(n_repeats):
        per_repeat[r] = rmse(oof_all[r], y)
    res = CVResult(name, float(per_repeat.mean()), float(per_repeat.std(ddof=1)),
                   float(per_repeat.std(ddof=1) / np.sqrt(n_repeats)),
                   per_repeat, oof_all[0], np.array(fold_rmses))
    if verbose:
        print("    " + res.line())
    return res


def loo_cv(name: str, est, df_raw: pd.DataFrame, y: np.ndarray,
           verbose: bool = True) -> tuple[float, np.ndarray]:
    """Leave-one-out. Worth running on 150 rows for the top two models."""
    y = np.asarray(y, float)
    oof = np.zeros(len(y))
    for tri, tei in LeaveOneOut().split(df_raw):
        e = clone(est)
        e.fit(df_raw.iloc[tri], y[tri])
        oof[tei] = e.predict(df_raw.iloc[tei])
    r = rmse(oof, y)
    if verbose:
        print(f"    {name:44s} LOO RMSE = {r:8.4f}")
    return r, oof


def nested_cv(name: str, make_est: Callable, search: Callable,
              df_raw: pd.DataFrame, y: np.ndarray, n_splits: int = 10,
              n_repeats: int = 3, seed: int = SEED, verbose: bool = True
              ) -> CVResult:
    """Outer loop scores; inner loop tunes. The reported number is unbiased.

    `search(df_tr, y_tr)` must run its own inner CV and return fitted
    hyperparameters; `make_est(best)` turns them into an estimator.
    """
    y = np.asarray(y, float)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                        random_state=seed)
    oof_all = np.zeros((n_repeats, len(y)))
    fold_rmses = []
    for k, (tri, tei) in enumerate(rkf.split(df_raw)):
        rep = k // n_splits
        best = search(df_raw.iloc[tri], y[tri])
        e = make_est(best)
        e.fit(df_raw.iloc[tri], y[tri])
        p = e.predict(df_raw.iloc[tei])
        oof_all[rep, tei] = p
        fold_rmses.append(rmse(p, y[tei]))
    per_repeat = np.array([rmse(oof_all[r], y) for r in range(n_repeats)])
    res = CVResult(name, float(per_repeat.mean()),
                   float(per_repeat.std(ddof=1)) if n_repeats > 1 else 0.0,
                   float(per_repeat.std(ddof=1) / np.sqrt(n_repeats))
                   if n_repeats > 1 else 0.0,
                   per_repeat, oof_all[0], np.array(fold_rmses))
    if verbose:
        print("    " + res.line() + "   [NESTED]")
    return res


# ==========================================================================
# Reporting
# ==========================================================================
def results_table(results: list[CVResult], baseline: float | None = None
                  ) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r.name,
            "CV_RMSE": r.mean_rmse,
            "sd_repeats": r.sd_repeats,
            "se": r.se_mean,
            "fold_rmse_min": r.fold_rmses.min(),
            "fold_rmse_max": r.fold_rmses.max(),
        })
    t = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    if baseline is not None:
        t["vs_baseline"] = t["CV_RMSE"] - baseline
    return t


def within_one_se(t: pd.DataFrame) -> pd.DataFrame:
    """Models statistically indistinguishable from the best.

    The brief's tie-break rule: if two models are within one standard error,
    take the simpler, more physically grounded one.
    """
    best = t.iloc[0]
    thr = best["CV_RMSE"] + best["se"]
    return t[t["CV_RMSE"] <= thr]


def paired_test(a: CVResult, b: CVResult) -> dict:
    """Paired comparison on identical repeats -- far sharper than two means."""
    from scipy import stats
    n = min(len(a.per_repeat), len(b.per_repeat))
    d = b.per_repeat[:n] - a.per_repeat[:n]
    t, p = stats.ttest_rel(b.per_repeat[:n], a.per_repeat[:n])
    return {"diff": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(n)), "p": float(p)}
