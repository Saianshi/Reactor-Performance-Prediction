"""Phase 1b -- follow-ups on the three surprises the first pass threw up.

  F1  CA0 looks completely inert. Confirm with a GP LOO comparison (4 vs 5
      inputs) and by looking for near-neighbour pairs that differ mainly in CA0.
  F2  37/150 rows are EXACTLY 0.000. Characterise that floor: is it the
      over-reaction regime (Da2 >> 1)?
  F3  A plain GP in raw coordinates does badly (LOO RMSE 17.6). Is the surface
      a sharp cliff in temperature, and does moving to Arrhenius/log
      coordinates tame it?
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from config import FEATURES, R_GAS, TARGET, TEST_CSV, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def gp_loo(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Exact leave-one-out RMSE of an interpolating Matern-5/2 ARD GP."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    ker = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
        length_scale=np.ones(X.shape[1]), nu=2.5, length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=4, random_state=seed)
    gp.fit(Xs, y)
    K = gp.kernel_(Xs)
    K[np.diag_indices_from(K)] += 1e-8
    ym, ys = y.mean(), y.std()
    Ki = np.linalg.inv(K)
    res = (Ki @ ((y - ym) / ys)) / np.diag(Ki) * ys
    return float(np.sqrt((res ** 2).mean()))


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    y = tr[TARGET].to_numpy(float)

    # ------------------------------------------------------------------
    rule("F1. IS CA0 COMPLETELY INERT?")
    no_ca = [f for f in FEATURES if f != "concentration_mol_L"]
    r5 = gp_loo(tr[FEATURES].to_numpy(float), y)
    r4 = gp_loo(tr[no_ca].to_numpy(float), y)
    print(f"    GP LOO RMSE with all 5 inputs   = {r5:.4f}")
    print(f"    GP LOO RMSE without CA0         = {r4:.4f}")
    print(f"    change from dropping CA0        = {r4 - r5:+.4f} "
          "(negative = dropping it HELPS)")

    # the GP's own ARD length-scale for CA0: a huge length-scale means the
    # kernel has switched that input off
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr[FEATURES])
    ker = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
        length_scale=np.ones(5), nu=2.5, length_scale_bounds=(1e-2, 1e4))
    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=6, random_state=0)
    gp.fit(sc.transform(tr[FEATURES]), y)
    ls = gp.kernel_.k2.length_scale
    print("\n    ARD length-scales (standardised units; large = input ignored):")
    for f, l in sorted(zip(FEATURES, ls), key=lambda t: t[1]):
        print(f"      {f:24s} {l:10.3f}"
              + ("   <-- switched OFF" if l > 50 else ""))

    # nearest-neighbour pairs in the OTHER four features
    print("\n    near-neighbour pairs matched on the other 4 inputs:")
    Z = StandardScaler().fit_transform(tr[no_ca])
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    iu = np.triu_indices(len(Z), 1)
    order = np.argsort(d[iu])[:8]
    print(f"    {'d4':>6} {'dCA0':>7} {'y_i':>8} {'y_j':>8} {'|dy|':>7}")
    dys, dcs = [], []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        dc = abs(tr.concentration_mol_L[i] - tr.concentration_mol_L[j])
        dy = abs(y[i] - y[j])
        dys.append(dy)
        dcs.append(dc)
        print(f"    {d[i, j]:6.3f} {dc:7.2f} {y[i]:8.3f} {y[j]:8.3f} {dy:7.3f}")
    print(f"    -> among the 8 closest 4-D pairs, mean |dCA0| = "
          f"{np.mean(dcs):.2f} mol/L but mean |dy| = {np.mean(dys):.3f} %")

    # ------------------------------------------------------------------
    rule("F2. THE ZERO FLOOR: 37/150 ROWS ARE EXACTLY 0.000")
    zero = y == 0
    print(f"    exact zeros: {zero.sum()}   (a stochastic plant would never "
          "repeat a value\n    exactly 37 times -> further proof this is a "
          "deterministic solver)")
    tau = (tr.length_m / tr.flow_rate_L_min).to_numpy()
    print("\n    what separates the zero rows? (mean of each group)")
    g = pd.DataFrame({**{f: tr[f] for f in FEATURES}, "tau": tau,
                      "zero": zero}).groupby("zero").mean()
    print(g.round(3).to_string())
    print("\n    a simple thermal severity index: Tj (the asymptote the fluid "
          "relaxes to)")
    for lo, hi in [(350, 400), (400, 430), (430, 460), (460, 500), (500, 550)]:
        m = (tr.jacket_temperature_K >= lo) & (tr.jacket_temperature_K < hi)
        if m.sum():
            print(f"      Tj in [{lo},{hi})  n={m.sum():3d}  "
                  f"mean yield={y[m].mean():6.2f}  "
                  f"frac exactly 0 = {zero[m].mean():.2f}")
    print("\n    the same cut on tau_proxy = L/Q:")
    qs = np.quantile(tau, [0, .2, .4, .6, .8, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (tau >= lo) & (tau <= hi)
        print(f"      tau in [{lo:.3f},{hi:.3f}]  n={m.sum():3d}  "
              f"mean yield={y[m].mean():6.2f}  frac 0 = {zero[m].mean():.2f}")

    # ------------------------------------------------------------------
    rule("F3. IS THE SURFACE A CLIFF? DOES ARRHENIUS GEOMETRY TAME IT?")
    print("    A first-order series reactor collapses onto a 2-D map:")
    print("      log(tau)  x  1/T   ->  Da1 = k1*tau  and  Da2 = k2*tau")
    print("    Try a GP in a few candidate coordinate systems (LOO RMSE):\n")

    Q = tr.flow_rate_L_min.to_numpy()
    L = tr.length_m.to_numpy()
    T0 = tr.inlet_temperature_K.to_numpy()
    Tj = tr.jacket_temperature_K.to_numpy()

    cands = {
        "raw 5 inputs": tr[FEATURES].to_numpy(float),
        "raw 4 (no CA0)": tr[no_ca].to_numpy(float),
        "log tau, Q, L, T0, Tj": np.c_[np.log(L / Q), np.log(Q), np.log(L), T0, Tj],
        "log tau, 1/T0, 1/Tj": np.c_[np.log(L / Q), 1 / T0, 1 / Tj],
        "log tau, 1/T0, 1/Tj, log Q": np.c_[np.log(L / Q), 1 / T0, 1 / Tj, np.log(Q)],
        "log tau, 1/T0, 1/Tj, log Q, log L": np.c_[
            np.log(L / Q), 1 / T0, 1 / Tj, np.log(Q), np.log(L)],
    }
    for name, Xc in cands.items():
        print(f"    {name:36s} d={Xc.shape[1]}  LOO RMSE = "
              f"{gp_loo(Xc, y):7.4f}")

    print("\n    and on a logit-like transform of the target "
          "(sqrt, to handle the 0 floor):")
    for name, Xc in list(cands.items())[:1] + list(cands.items())[-1:]:
        print(f"    {name:36s} sqrt-target LOO RMSE (back-transformed) = "
              f"{gp_loo(Xc, np.sqrt(y)) ** 2:7.4f}  "
              "(not directly comparable, indicative only)")

    # ------------------------------------------------------------------
    rule("F4. HOW MUCH OF THE TEST SET SITS IN THE DEAD ZONE?")
    tau_te = (te.length_m / te.flow_rate_L_min).to_numpy()
    print(f"    train: {100 * zero.mean():.1f}% of rows are exactly zero")
    print(f"    train Tj mean {Tj.mean():.1f} K   test Tj mean "
          f"{te.jacket_temperature_K.mean():.1f} K")
    print(f"    train frac Tj>480K = "
          f"{(Tj > 480).mean():.2f}   test frac Tj>480K = "
          f"{(te.jacket_temperature_K > 480).mean():.2f}")
    print(f"    train tau median {np.median(tau):.3f}   test tau median "
          f"{np.median(tau_te):.3f}")
    print("\n    -> we should expect a similar share of near-zero true values "
          "in the\n    hidden test labels. Getting the zeros exactly right is "
          "worth a lot of RMSE.")


if __name__ == "__main__":
    main()
