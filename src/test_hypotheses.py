"""Phase 2b driver -- adjudicate H1 (thermal) vs H2 (kinetic order) vs H3
(mixing) as explanations for the real, proven CA0 dependence.

Reported for each: training RMSE, 10-fold CV RMSE with the physics REFITTED
inside every fold, parameter count, and the fitted value of the parameter that
carries the hypothesis. The winner is the one that generalises, not the one
that fits.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import physics as ph
import physics_ext as pe
from config import CV_DIR, SEED, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def rule(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


def cv_generic(df, y, fit_fn, pred_fn, n_splits=10, seed=SEED):
    """Refit inside every fold. Returns (pooled OOF rmse, oof, param frame)."""
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    for i, (tri, tei) in enumerate(kf.split(df)):
        p, r = fit_fn(df.iloc[tri], y[tri])
        oof[tei] = pred_fn(p, df.iloc[tei])
        d = {k: getattr(p, k) for k in
             (pe.EXT_NAMES if isinstance(p, pe.ExtParams) else ph.PARAM_NAMES)}
        d["fold"] = i
        rows.append(d)
    return ph.rmse(oof, y), oof, pd.DataFrame(rows)


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)

    results = []

    # ------------------------------------------------------------------
    rule("H0 / H1  BASE MODEL: first order, plug flow, thermal term only")
    for label, fixed, k in [
            ("H0 gamma=0 (no CA0 effect at all)", {"n_flow": 0.0, "gamma": 0.0}, 5),
            ("H1 gamma free (thermal)          ", {"n_flow": 0.0}, 6)]:
        t0 = time.time()
        p, r, _ = ph.fit_physics(tr, y, level="B", fixed=fixed, seed=SEED,
                                 maxiter=60, popsize=12, verbose=False)

        def fit_fn(d, yy, _f=fixed):
            pp, rr, _ = ph.fit_physics(d, yy, level="B", fixed=_f, seed=SEED,
                                       maxiter=60, popsize=12, verbose=False)
            return pp, rr

        cv, _, P = cv_generic(tr, y, fit_fn,
                              lambda pp, d: ph.level_b(pp, d, n_steps=400))
        print(f"    {label}  k={k}  train={r:7.4f}  CV={cv:7.4f}  "
              f"({time.time() - t0:.0f}s)")
        print(f"        Ea1={p.Ea1/1e3:6.1f}  Ea2={p.Ea2/1e3:6.1f} kJ/mol  "
              f"beta0={p.beta0:7.3f}  gamma={p.gamma:+7.3f}")
        print(f"        gamma across folds: {P.gamma.mean():+.3f} +/- "
              f"{P.gamma.std():.3f}")
        results.append({"hypothesis": label.strip(), "k": k, "train": r,
                        "cv": cv, "carrier": f"gamma={p.gamma:+.3f}"})

    # ------------------------------------------------------------------
    rule("H2  KINETIC ORDER: rate1 = k1*CA^n1, rate2 = k2*CB^n2")
    print("    If n1 = 1 the concentration cancels exactly. A fitted n1 that")
    print("    differs from 1 IS the CA0 dependence, with no thermal story.\n")
    for label, fixed, k in [
            ("H2a n1 free, n2=1, gamma=0 ", {"n_flow": 0.0, "gamma": 0.0,
                                             "n2": 1.0}, 6),
            ("H2b n1,n2 free, gamma=0    ", {"n_flow": 0.0, "gamma": 0.0}, 7),
            ("H2c n1,n2 free, gamma free ", {"n_flow": 0.0}, 8)]:
        t0 = time.time()
        p, r = pe.fit_ext(tr, y, "orders", fixed=fixed, seed=SEED)

        def fit_fn(d, yy, _f=fixed):
            return pe.fit_ext(d, yy, "orders", fixed=_f, seed=SEED)

        cv, _, P = cv_generic(tr, y, fit_fn,
                              lambda pp, d: pe.level_b_orders(pp, d, n_steps=400))
        print(f"    {label}  k={k}  train={r:7.4f}  CV={cv:7.4f}  "
              f"({time.time() - t0:.0f}s)")
        print(f"        n1={p.n1:.4f}  n2={p.n2:.4f}  Ea1={p.Ea1/1e3:6.1f}  "
              f"Ea2={p.Ea2/1e3:6.1f} kJ/mol  gamma={p.gamma:+7.3f}")
        print(f"        n1 across folds: {P.n1.mean():.4f} +/- {P.n1.std():.4f}"
              f"   n2: {P.n2.mean():.4f} +/- {P.n2.std():.4f}")
        results.append({"hypothesis": label.strip(), "k": k, "train": r,
                        "cv": cv, "carrier": f"n1={p.n1:.3f},n2={p.n2:.3f}"})

    # ------------------------------------------------------------------
    rule("H3  MIXING: tanks in series (the algebraic form of axial dispersion)")
    print("    The organisers called it a BOUNDARY value problem. A plug flow")
    print("    reactor is an INITIAL value problem; axial dispersion is what")
    print("    turns it into a BVP. N tanks <-> Peclet ~ 2(N-1).\n")
    print(f"    {'N tanks':>9}{'Pe~2(N-1)':>11}{'train RMSE':>13}"
          f"{'CV RMSE':>11}{'gamma':>9}")
    print("    " + "-" * 54)
    for N in [1, 2, 3, 5, 10, 20, 50]:
        fixed = {"n_flow": 0.0}
        p, r = pe.fit_ext(tr, y, "tanks", fixed=fixed, seed=SEED, n_tanks=N)

        def fit_fn(d, yy, _N=N, _f=fixed):
            return pe.fit_ext(d, yy, "tanks", fixed=_f, seed=SEED, n_tanks=_N)

        cv, _, _ = cv_generic(tr, y, fit_fn,
                              lambda pp, d, _N=N: pe.level_cstr_series(
                                  pp, d, n_tanks=_N))
        print(f"    {N:9d}{2 * (N - 1):11d}{r:13.4f}{cv:11.4f}{p.gamma:+9.3f}")
        results.append({"hypothesis": f"H3 tanks N={N}", "k": 6, "train": r,
                        "cv": cv, "carrier": f"N={N}"})

    # ------------------------------------------------------------------
    rule("VERDICT")
    t = pd.DataFrame(results).sort_values("cv").reset_index(drop=True)
    print(t.round(4).to_string(index=False))
    t.to_csv(CV_DIR / "hypothesis_comparison.csv", index=False)
    print(f"\n    baseline (predict the mean) = {y.std():.3f}")
    best = t.iloc[0]
    print(f"\n    best by CV: {best.hypothesis}  (CV {best.cv:.4f}, "
          f"train {best.train:.4f}, {best.carrier})")


if __name__ == "__main__":
    main()
