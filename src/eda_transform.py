"""Phase 1c -- target transform screening, correctly back-transformed.

RMSE is always reported in the ORIGINAL yield-% units: predict in transformed
space, invert the transform, then score. Anything else is not comparable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import FEATURES, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")


def gp_loo_pred(X: np.ndarray, z: np.ndarray, seed: int = 0) -> np.ndarray:
    """Exact LOO predictions of an interpolating Matern-5/2 ARD GP."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    ker = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
        length_scale=np.ones(X.shape[1]), nu=2.5,
        length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=4, random_state=seed)
    gp.fit(Xs, z)
    K = gp.kernel_(Xs)
    K[np.diag_indices_from(K)] += 1e-8
    m, s = z.mean(), z.std()
    Ki = np.linalg.inv(K)
    res = (Ki @ ((z - m) / s)) / np.diag(Ki) * s
    return z - res


EPS = 1e-3


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)
    Q = tr.flow_rate_L_min.to_numpy()
    L = tr.length_m.to_numpy()
    T0 = tr.inlet_temperature_K.to_numpy()
    Tj = tr.jacket_temperature_K.to_numpy()
    CA = tr.concentration_mol_L.to_numpy()

    coords = {
        "raw 5 inputs": tr[FEATURES].to_numpy(float),
        "log tau, log Q, log L, T0, Tj": np.c_[np.log(L / Q), np.log(Q),
                                               np.log(L), T0, Tj],
        "log tau, log Q, 1/T0, 1/Tj, CA0": np.c_[np.log(L / Q), np.log(Q),
                                                 1 / T0, 1 / Tj, CA],
    }

    p = y / 100.0
    transforms = {
        "identity        ": (lambda v: v, lambda z: z),
        "sqrt            ": (lambda v: np.sqrt(v), lambda z: np.clip(z, 0, None) ** 2),
        "cbrt            ": (lambda v: np.cbrt(v), lambda z: z ** 3),
        "logit(eps-clip) ": (
            lambda v: np.log(np.clip(v / 100, EPS, 1 - EPS)
                             / (1 - np.clip(v / 100, EPS, 1 - EPS))),
            lambda z: 100 / (1 + np.exp(-z))),
        "log1p           ": (lambda v: np.log1p(v), lambda z: np.expm1(z)),
    }

    print("=" * 90)
    print("TARGET TRANSFORM SCREEN -- GP leave-one-out, RMSE in yield-% "
          "after back-transform")
    print("=" * 90)
    print(f"    baseline: predict the mean          RMSE = {y.std():.4f}")
    print(f"    fraction of rows exactly 0.000      = {(y == 0).mean():.3f}\n")

    hdr = f"{'coordinates':34s}" + "".join(f"{t.strip():>12s}" for t in transforms)
    print(hdr)
    print("-" * len(hdr))
    for cname, X in coords.items():
        row = f"{cname:34s}"
        for tname, (fwd, inv) in transforms.items():
            z = fwd(y)
            pred = np.clip(inv(gp_loo_pred(X, z)), 0, 100)
            row += f"{np.sqrt(((pred - y) ** 2).mean()):12.4f}"
        print(row)

    print("\nNOTE: clipping predictions to [0,100] is applied in every case, "
          "so the\ncomparison isolates the transform, not the clipping.")

    # how much does clipping alone buy on the identity model?
    X = coords["log tau, log Q, log L, T0, Tj"]
    raw = gp_loo_pred(X, y)
    print(f"\nclipping alone, identity target, best coords:")
    print(f"    unclipped RMSE = {np.sqrt(((raw - y) ** 2).mean()):.4f}")
    print(f"    clipped   RMSE = "
          f"{np.sqrt(((np.clip(raw, 0, 100) - y) ** 2).mean()):.4f}")
    print(f"    predictions below 0: {(raw < 0).sum()}, above 100: "
          f"{(raw > 100).sum()}")


if __name__ == "__main__":
    main()
