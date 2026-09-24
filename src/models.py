"""Phase 4 -- the model zoo.

Every model is a factory returning an UNFITTED sklearn-compatible estimator,
so that validate.py can clone and refit it inside each fold. No model here
ever sees the full dataset.

The headline design is `gp_residual`: a Gaussian process whose prior mean is
the fitted physics prediction. That is the standard surrogate for
deterministic simulator output -- it interpolates, it returns calibrated
uncertainty, and because the physics carries the shape of the response
surface, the GP only has to learn a small, smooth correction.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              RationalQuadratic, WhiteKernel)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED


# ==========================================================================
# Gaussian processes
# ==========================================================================
class ScaledGP(BaseEstimator, RegressorMixin):
    """GP with input standardisation fitted INSIDE the estimator.

    Keeping the scaler here (rather than outside) is what guarantees it is
    refitted on the fold's training rows only.
    """

    def __init__(self, nu: float = 2.5, alpha: float = 1e-8,
                 n_restarts: int = 8, white: bool = True, seed: int = SEED,
                 kernel: str = "matern"):
        self.nu = nu
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.white = white
        self.seed = seed
        self.kernel = kernel

    def _make_kernel(self, d):
        amp = ConstantKernel(1.0, (1e-4, 1e6))
        if self.kernel == "matern":
            base = Matern(length_scale=np.ones(d), nu=self.nu,
                          length_scale_bounds=(1e-2, 1e3))
        elif self.kernel == "rq":
            base = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                     length_scale_bounds=(1e-2, 1e3))
        else:
            raise ValueError(self.kernel)
        k = amp * base
        if self.white:
            # bounded well below the target scale: this is a numerical nugget
            # for a deterministic simulator, not a noise model
            k = k + WhiteKernel(1e-6, (1e-12, 1e-1))
        return k

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.scaler_ = StandardScaler().fit(X)
        self.gp_ = GaussianProcessRegressor(
            kernel=self._make_kernel(X.shape[1]), alpha=self.alpha,
            normalize_y=True, n_restarts_optimizer=self.n_restarts,
            random_state=self.seed)
        self.gp_.fit(self.scaler_.transform(X), y)
        return self

    def predict(self, X, return_std: bool = False):
        Xs = self.scaler_.transform(np.asarray(X, float))
        return self.gp_.predict(Xs, return_std=return_std)


def gp_matern(**kw):
    return ScaledGP(nu=2.5, **kw)


def gp_matern32(**kw):
    return ScaledGP(nu=1.5, **kw)


def gp_rq(**kw):
    return ScaledGP(kernel="rq", **kw)


# ==========================================================================
# Kernel ridge
# ==========================================================================
def krr(alpha: float = 1e-3, gamma: float | None = None):
    return Pipeline([("sc", StandardScaler()),
                     ("m", KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma))])


# ==========================================================================
# Trees
# ==========================================================================
def extra_trees(n_estimators: int = 800, **kw):
    return ExtraTreesRegressor(n_estimators=n_estimators, random_state=SEED,
                               n_jobs=-1, **kw)


def random_forest(n_estimators: int = 800, **kw):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=SEED,
                                 n_jobs=-1, **kw)


def xgb(device: str = "cpu", **kw):
    from xgboost import XGBRegressor
    p = dict(n_estimators=1200, learning_rate=0.03, max_depth=3,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
             min_child_weight=2, random_state=SEED, device=device,
             tree_method="hist", verbosity=0)
    p.update(kw)
    return XGBRegressor(**p)


def lgbm(device: str = "cpu", **kw):
    from lightgbm import LGBMRegressor
    p = dict(n_estimators=1200, learning_rate=0.03, num_leaves=7,
             min_child_samples=5, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED,
             device_type=device, verbose=-1)
    p.update(kw)
    return LGBMRegressor(**p)


def catb(task_type: str = "CPU", **kw):
    from catboost import CatBoostRegressor
    p = dict(iterations=1500, learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
             random_seed=SEED, task_type=task_type, verbose=0,
             allow_writing_files=False)
    p.update(kw)
    return CatBoostRegressor(**p)


# ==========================================================================
# Neural net
# ==========================================================================
def mlp(hidden=(64, 64), alpha: float = 1e-3, max_iter: int = 4000):
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                           learning_rate_init=3e-3, max_iter=max_iter,
                           early_stopping=True, n_iter_no_change=50,
                           validation_fraction=0.15, random_state=SEED))])


# ==========================================================================
# Stacking by non-negative least squares
# ==========================================================================
class NNLSStack(BaseEstimator, RegressorMixin):
    """Combine out-of-fold predictions with non-negative weights summing to 1.

    Non-negativity and the sum-to-one constraint are what stop the stack
    extrapolating wildly outside the range its members produce -- important
    here because the target is physically bounded.
    """

    def __init__(self, intercept: bool = False):
        self.intercept = intercept

    def fit(self, P, y):
        from scipy.optimize import nnls
        P = np.asarray(P, float)
        y = np.asarray(y, float)
        # append a row enforcing sum(w) = 1 with a large weight
        big = 1e3
        A = np.vstack([P, big * np.ones((1, P.shape[1]))])
        b = np.concatenate([y, [big]])
        w, _ = nnls(A, b)
        self.w_ = w
        return self

    def predict(self, P):
        return np.asarray(P, float) @ self.w_
