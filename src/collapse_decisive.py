"""Phase 2a(ii) -- decisive, error-barred adjudication of two questions.

The first pass left a contradiction:
  * T1 said adding CA0 to the 3-D set improved LOO from 16.32 to 14.55.
  * T2 said a GP could extract NOTHING about the residual from CA0.
A single LOO number has no error bar, so neither claim is yet trustworthy.

Here we use RepeatedKFold(10 x 5) with the GP refitted inside every fold, and
report PAIRED differences with a standard error across repeats. Paired, because
the fold-to-fold variance is large and common to all coordinate sets; the
paired difference is far more sensitive than comparing two noisy means.

Q1  Do L and Q act only through tau = L/Q?          (the collapse)
Q2  Does CA0 carry CONDITIONAL information given (tau, T0, Tj)?
    Note this is not contradicted by Phase 1: MI(CA0; y) = 0 is a MARGINAL
    statement. An exothermic heat-release term T_rise = gamma*CA0*X produces
    exactly this signature -- no marginal effect, but a real conditional one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from config import CV_DIR, FEATURES, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

N_SPLITS, N_REPEATS, RESTARTS = 10, 5, 6


def rule(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


def make_gp(d, seed=0):
    ker = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
        length_scale=np.ones(d), nu=2.5, length_scale_bounds=(1e-2, 1e3))
    return GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                    n_restarts_optimizer=RESTARTS,
                                    random_state=seed)


def repeated_cv_preds(X, y, seed=0):
    """Return an (n_repeats, n) array of out-of-fold predictions.

    Scaler AND GP are fitted inside each fold. Predictions clipped to [0,100].
    """
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                        random_state=seed)
    out = np.zeros((N_REPEATS, len(y)))
    for k, (tri, tei) in enumerate(rkf.split(X)):
        rep = k // N_SPLITS
        sc = StandardScaler().fit(X[tri])
        gp = make_gp(X.shape[1], seed=0)
        gp.fit(sc.transform(X[tri]), y[tri])
        out[rep, tei] = np.clip(gp.predict(sc.transform(X[tei])), 0, 100)
    return out


def rmse_per_repeat(preds, y):
    return np.sqrt(((preds - y[None, :]) ** 2).mean(1))


def coord_sets(df):
    Q = df.flow_rate_L_min.to_numpy(float)
    L = df.length_m.to_numpy(float)
    T0 = df.inlet_temperature_K.to_numpy(float)
    Tj = df.jacket_temperature_K.to_numpy(float)
    CA = df.concentration_mol_L.to_numpy(float)
    tau = L / Q
    return {
        "A  logtau,T0,Tj              [3]": np.c_[np.log(tau), T0, Tj],
        "B  logtau,T0,Tj,logQ         [4]": np.c_[np.log(tau), T0, Tj, np.log(Q)],
        "C  logtau,T0,Tj,CA0          [4]": np.c_[np.log(tau), T0, Tj, CA],
        "D  logtau,T0,Tj,logQ,CA0     [5]": np.c_[np.log(tau), T0, Tj,
                                                  np.log(Q), CA],
        "E  logQ,logL,T0,Tj           [4]": np.c_[np.log(Q), np.log(L), T0, Tj],
        "F  logQ,logL,T0,Tj,CA0       [5]": np.c_[np.log(Q), np.log(L), T0, Tj,
                                                  CA],
        "G  logtau,1/T0,1/Tj          [3]": np.c_[np.log(tau), 1 / T0, 1 / Tj],
        "H  logtau,1/T0,1/Tj,CA0      [4]": np.c_[np.log(tau), 1 / T0, 1 / Tj,
                                                  CA],
        "I  raw 5 inputs              [5]": df[FEATURES].to_numpy(float),
    }


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)
    sets = coord_sets(tr)

    rule(f"REPEATED CV: {N_SPLITS}-fold x {N_REPEATS} repeats, GP refitted "
         "in every fold")
    print("    Scaler and GP both fitted inside the fold. Clipped to [0,100].")
    print(f"    Baseline (predict fold-train mean) ~ {y.std():.2f}\n")

    res = {}
    print(f"    {'coordinate set':36s}{'mean':>9}{'sd':>8}   per-repeat RMSE")
    print("    " + "-" * 88)
    for name, X in sets.items():
        p = repeated_cv_preds(X, y)
        r = rmse_per_repeat(p, y)
        res[name] = {"preds": p, "rmse": r}
        print(f"    {name:36s}{r.mean():9.3f}{r.std():8.3f}   "
              + " ".join(f"{v:6.2f}" for v in r))

    tbl = pd.DataFrame({k: v["rmse"] for k, v in res.items()})
    tbl.to_csv(CV_DIR / "collapse_repeatedcv.csv", index=False)

    # ------------------------------------------------------------------
    rule("PAIRED COMPARISONS (same folds, same seed -> paired is much sharper)")

    def paired(a, b, question):
        ra, rb = res[a]["rmse"], res[b]["rmse"]
        d = rb - ra           # positive => b is WORSE => a is better
        se = d.std(ddof=1) / np.sqrt(len(d))
        t, p = stats.ttest_rel(rb, ra)
        print(f"\n    {question}")
        print(f"      {a.strip()}   -> {ra.mean():.3f}")
        print(f"      {b.strip()}   -> {rb.mean():.3f}")
        print(f"      difference (b - a) = {d.mean():+.3f} +/- {se:.3f} (se)"
              f"   paired t p = {p:.4f}")
        verdict = ("no detectable difference" if p > 0.05 else
                   (f"{b.strip().split()[0]} is WORSE" if d.mean() > 0
                    else f"{b.strip().split()[0]} is BETTER"))
        print(f"      -> {verdict}")
        return d.mean(), se, p

    print("\n  Q1. THE COLLAPSE: do L and Q act only through tau?")
    paired("A  logtau,T0,Tj              [3]",
           "B  logtau,T0,Tj,logQ         [4]",
           "adding log Q on top of tau (if tau suffices, no gain):")
    paired("A  logtau,T0,Tj              [3]",
           "E  logQ,logL,T0,Tj           [4]",
           "replacing tau by (log Q, log L) separately:")
    paired("C  logtau,T0,Tj,CA0          [4]",
           "F  logQ,logL,T0,Tj,CA0       [5]",
           "same, with CA0 present in both:")

    print("\n  Q2. DOES CA0 CARRY CONDITIONAL INFORMATION?")
    paired("C  logtau,T0,Tj,CA0          [4]",
           "A  logtau,T0,Tj              [3]",
           "removing CA0 from the collapsed 3-D set:")
    paired("H  logtau,1/T0,1/Tj,CA0      [4]",
           "G  logtau,1/T0,1/Tj          [3]",
           "same, in Arrhenius 1/T coordinates:")

    print("\n  Q3. CO-ORDINATES: raw T or Arrhenius 1/T?")
    paired("A  logtau,T0,Tj              [3]",
           "G  logtau,1/T0,1/Tj          [3]",
           "1/T instead of T (over this 150 K span the two are near-affine):")
    paired("D  logtau,T0,Tj,logQ,CA0     [5]",
           "I  raw 5 inputs              [5]",
           "physics coordinates vs raw inputs:")

    # ------------------------------------------------------------------
    rule("PERMUTATION TEST: is the CA0 gain bigger than chance?")
    print("    Refit set C with CA0 replaced by a random shuffle of itself.")
    print("    If the real CA0 gain is genuine it must beat the shuffled gain.\n")
    X_A = sets["A  logtau,T0,Tj              [3]"]
    r_A = res["A  logtau,T0,Tj              [3]"]["rmse"].mean()
    r_C = res["C  logtau,T0,Tj,CA0          [4]"]["rmse"].mean()
    CA = tr.concentration_mol_L.to_numpy(float)
    rng = np.random.default_rng(0)
    shuf = []
    for i in range(8):
        Xs = np.c_[X_A, rng.permutation(CA)]
        shuf.append(rmse_per_repeat(repeated_cv_preds(Xs, y), y).mean())
        print(f"      shuffle {i + 1}: RMSE = {shuf[-1]:7.3f}  "
              f"(gain vs 3-D = {r_A - shuf[-1]:+.3f})")
    shuf = np.array(shuf)
    print(f"\n      real CA0     : RMSE = {r_C:7.3f}  "
          f"(gain vs 3-D = {r_A - r_C:+.3f})")
    print(f"      shuffled CA0 : RMSE = {shuf.mean():7.3f} +/- {shuf.std():.3f}"
          f"  (mean gain = {r_A - shuf.mean():+.3f})")
    z = (shuf.mean() - r_C) / (shuf.std(ddof=1) + 1e-12)
    print(f"      real gain is {z:.2f} shuffle-sd better than chance")
    print(f"      -> {'CA0 CARRIES REAL CONDITIONAL SIGNAL' if z > 2 else 'CA0 gain is within chance'}")

    # ------------------------------------------------------------------
    rule("ARD LENGTH-SCALES ON THE FULL 5-D PHYSICS SET (what the GP ignores)")
    X = sets["D  logtau,T0,Tj,logQ,CA0     [5]"]
    sc = StandardScaler().fit(X)
    gp = make_gp(5, seed=0)
    gp.fit(sc.transform(X), y)
    names = ["log tau", "T0", "Tj", "log Q", "CA0"]
    ls = gp.kernel_.k2.length_scale
    print("    (standardised units; a length-scale at the 1e3 bound = ignored)")
    for n, l in sorted(zip(names, ls), key=lambda t: t[1]):
        tag = "   <-- SWITCHED OFF by ARD" if l > 100 else ""
        print(f"      {n:10s} {l:12.4f}{tag}")


if __name__ == "__main__":
    main()
