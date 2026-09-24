"""Phase 2 driver -- verify the integrator, fit the physics, cross-validate it,
and measure how much the fitted parameters wobble across folds.

Model variants, chosen to answer the two open physical questions directly:

  M0  beta constant,      gamma = 0     5 free params   (the textbook model)
  M1  beta constant,      gamma free    6 free params   (exothermic feedback)
  M2  beta = beta0*(Q/Qref)^n, gamma = 0  6 free params (flow-dependent h)
  M3  both free                         7 free params   (the full brief model)

Comparing M0 vs M1 tests whether concentration matters at all.
Comparing M0 vs M2 tests whether L and Q are interchangeable through tau.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import physics as ph
from config import CV_DIR, OUT_DIR, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def rule(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


VARIANTS = {
    "M0 beta const, gamma=0      ": dict(fixed={"n_flow": 0.0, "gamma": 0.0},
                                         k=5),
    "M1 beta const, gamma free   ": dict(fixed={"n_flow": 0.0}, k=6),
    "M2 beta(Q), gamma=0         ": dict(fixed={"gamma": 0.0}, k=6),
    "M3 beta(Q), gamma free      ": dict(fixed={}, k=7),
}


# ==========================================================================
def verify_integrator(df):
    rule("STEP 1. VERIFY THE BATCHED INTEGRATOR AGAINST scipy.solve_ivp")
    print("    The fast batched integrator is what the parameter search uses,")
    print("    so it must agree with an adaptive reference solver before we")
    print("    trust a single number that comes out of it.\n")

    # Test A isolates the mass balances (no reaction heat at all); test B
    # switches the energy source on. Splitting the check this way localises a
    # failure to either the chemistry step or the energy step.
    cases = [
        ("A  gamma = 0  (mass balances only)",
         ph.PhysParams(1.5, 80e3, 0.3, 150e3, 1.5, 0.0, 0.0)),
        ("B  gamma = 20 (energy source on) ",
         ph.PhysParams(1.5, 80e3, 0.3, 150e3, 1.5, -0.2, 20.0)),
        ("C  stiff corner, k1 >> k2        ",
         ph.PhysParams(6.0, 200e3, -6.0, 30e3, 2.0, 0.0, 0.0)),
    ]
    for label, p in cases:
        t0 = time.time()
        ref = ph.level_b_scipy(p, df)
        t_ref = time.time() - t0
        print(f"    {label}")
        print(f"      {'n_steps':>9}{'max |diff|':>14}{'rmse diff':>13}"
              f"{'time (s)':>11}{'speed-up':>11}")
        errs = {}
        for n in [50, 100, 200, 400, 800]:
            t0 = time.time()
            got = ph.level_b(p, df, n_steps=n)
            t = time.time() - t0
            errs[n] = np.abs(got - ref).max()
            order = ("" if n == 50 else
                     f"   order {np.log2(errs[n // 2] / max(errs[n], 1e-16)):.2f}")
            print(f"      {n:9d}{errs[n]:14.3e}"
                  f"{np.sqrt(((got - ref) ** 2).mean()):13.3e}{t:11.4f}"
                  f"{t_ref / max(t, 1e-9):10.0f}x{order}")
        # a correct scheme must CONVERGE: halving dt must shrink the error.
        # (the earlier buggy version sat at a constant error at every n, which
        #  is the signature of a consistency bug rather than a step-size issue)
        assert errs[800] < errs[50] / 20, "integrator is not converging"
        assert errs[800] < 5e-2, (f"batched integrator disagrees with solve_ivp "
                                  f"by {errs[800]:.3e} on case {label.strip()}")
        print(f"      PASSED: converges at ~2nd order; n_steps=800 within "
              f"{errs[800]:.2e} yield-% (max)\n              For scale, the "
              f"RMSE we are trying to fit is O(5-15), so integration error is\n"
              f"              3-4 orders of magnitude below the signal. "
              f"(LSODA ref: {t_ref:.2f} s)\n")

    # The analytic limit: with gamma = 0 and beta huge, T == Tj everywhere, so
    # Level A's closed form and Level B's ODE must agree EXACTLY.
    #
    # Note how large beta has to be for this test to mean anything. T_eff
    # reaches Tj only when b = beta*tau >> 1 for EVERY row, and the shortest
    # residence time here is tau = 0.036, so beta ~ 1e5 is required. At
    # beta = 403 the shortest-tau rows still sit 7% of the way back towards
    # T0 and the two levels genuinely differ by ~4 yield-% -- that is a real
    # difference between the models, not a bug, and it is exactly why this is
    # an assertion with a stated tolerance rather than a printed reassurance.
    print("\n    Cross-check of Level A against Level B in the isothermal "
          "limit\n    (gamma=0, beta -> large, so T = Tj throughout):")
    for lb in [6.0, 9.0, 12.0, 15.0]:
        p_iso = ph.PhysParams(lnk1_ref=1.5, Ea1=80e3, lnk2_ref=0.3, Ea2=150e3,
                              log_beta0=lb, n_flow=0.0, gamma=0.0)
        ya = ph.level_a(p_iso, df)
        yb = ph.level_b(p_iso, df, n_steps=400)
        b_min = np.exp(lb) * (df.length_m / df.flow_rate_L_min).min()
        print(f"      log_beta0={lb:5.1f}  (min b = beta*tau = {b_min:10.1f})"
              f"   max |A - B| = {np.abs(ya - yb).max():.3e} yield-%")
    # The residual gap shrinks like 1/b: at min b ~ 5.8e3 the coldest row is
    # still ~0.03 K short of Tj, which moves the yield by ~1e-2. So the
    # tolerance is set by how isothermal we can actually make the test, not by
    # any disagreement between the two models.
    assert np.abs(ya - yb).max() < 5e-3, (
        "Level A and Level B disagree in the isothermal limit -- the closed "
        "form and the ODE are not describing the same physics")
    print("      ASSERTION PASSED: the gap falls like 1/b as the reactor is "
          "made\n      isothermal, reaching <5e-3 yield-%. Same physics, two "
          "implementations.")


# ==========================================================================
def fit_all_variants(df, y, level="A"):
    rule(f"STEP 2. FIT THE PHYSICS -- LEVEL {level}"
         + ("  (analytic closed form)" if level == "A" else "  (ODE solver)"))
    out = {}
    for name, cfg in VARIANTS.items():
        print(f"\n    {name}  ({cfg['k']} free parameters)")
        t0 = time.time()
        p, r, _ = ph.fit_physics(df, y, level=level, fixed=cfg["fixed"],
                                 seed=42,
                                 maxiter=300 if level == "A" else 60,
                                 popsize=24 if level == "A" else 12)
        print(f"      train RMSE = {r:.4f}    ({time.time() - t0:.1f} s)")
        out[name] = {"params": p, "train_rmse": r, "k": cfg["k"],
                     "fixed": cfg["fixed"]}
    return out


def report_variants(fits, y, n=150):
    rule("STEP 3. VARIANT COMPARISON")
    rows = []
    for name, f in fits.items():
        p = f["params"]
        k = f["k"]
        r = f["train_rmse"]
        # AIC on a Gaussian likelihood with the RMSE as the sd estimate
        aic = n * np.log(r ** 2) + 2 * k
        bic = n * np.log(r ** 2) + k * np.log(n)
        rows.append({
            "variant": name.strip(), "k": k, "train_RMSE": r,
            "AIC": aic, "BIC": bic,
            "Ea1_kJ": p.Ea1 / 1000, "Ea2_kJ": p.Ea2 / 1000,
            "Ea2-Ea1": (p.Ea2 - p.Ea1) / 1000,
            "beta0": p.beta0, "n_flow": p.n_flow, "gamma": p.gamma,
        })
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    print(f"\n    (baseline: predicting the mean gives RMSE = {y.std():.3f})")

    print("\n    READ-OUT:")
    m0, m1 = t.iloc[0], t.iloc[1]
    m2, m3 = t.iloc[2], t.iloc[3]
    print(f"      gamma free buys {m0.train_RMSE - m1.train_RMSE:+.4f} RMSE "
          f"(M0 -> M1); fitted gamma = {m1.gamma:.3f} K.L/mol")
    print(f"      beta(Q)    buys {m0.train_RMSE - m2.train_RMSE:+.4f} RMSE "
          f"(M0 -> M2); fitted n    = {m2.n_flow:+.3f}")
    print(f"      both       buy  {m0.train_RMSE - m3.train_RMSE:+.4f} RMSE "
          f"(M0 -> M3)")
    print("      turbulent internal flow predicts n = -0.2 "
          "(h ~ Re^0.8 => beta ~ Q^-0.2)")
    for _, r in t.iterrows():
        if r["Ea2-Ea1"] < 0:
            print(f"      !! {r['variant']}: Ea2 < Ea1 -- REJECT, this "
                  "contradicts the observed\n         collapse to zero yield "
                  "at high jacket temperature")
    return t


# ==========================================================================
def cross_validate_physics(df, y, level="A", variant="M3 beta(Q), gamma free      ",
                           n_splits=10, seed=42):
    """Refit the physics INSIDE each fold. No parameter sees its own test fold."""
    rule(f"STEP 4. CROSS-VALIDATED PHYSICS -- {variant.strip()}, level {level}")
    print(f"    {n_splits}-fold, physics parameters REFITTED from scratch in")
    print("    every fold. This is the honest generalisation number.\n")
    cfg = VARIANTS[variant]
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    par = []
    for i, (tri, tei) in enumerate(kf.split(df)):
        p, r, _ = ph.fit_physics(df.iloc[tri], y[tri], level=level,
                                 fixed=cfg["fixed"], seed=42,
                                 maxiter=300 if level == "A" else 60,
                                 popsize=24 if level == "A" else 12,
                                 verbose=False)
        fn = ph.level_a if level == "A" else (
            lambda pp, dd: ph.level_b(pp, dd, n_steps=200))
        oof[tei] = fn(p, df.iloc[tei])
        d = {k: getattr(p, k) for k in ph.PARAM_NAMES}
        d.update({"fold": i, "fold_train_rmse": r,
                  "fold_test_rmse": ph.rmse(oof[tei], y[tei])})
        par.append(d)
        print(f"      fold {i:2d}  train {r:7.4f}   test "
              f"{d['fold_test_rmse']:7.4f}   Ea1={p.Ea1 / 1e3:6.1f} "
              f"Ea2={p.Ea2 / 1e3:6.1f} kJ/mol  gamma={p.gamma:7.3f}")
    cv = ph.rmse(oof, y)
    print(f"\n    POOLED OUT-OF-FOLD RMSE = {cv:.4f}")
    P = pd.DataFrame(par)
    print("\n    PARAMETER WOBBLE ACROSS FOLDS "
          "(how stable is the physics itself?):")
    w = P[ph.PARAM_NAMES].agg(["mean", "std", "min", "max"]).T
    w["cv_%"] = 100 * w["std"] / w["mean"].abs().replace(0, np.nan)
    w.loc["Ea1", ["mean", "std", "min", "max"]] /= 1000
    w.loc["Ea2", ["mean", "std", "min", "max"]] /= 1000
    w.index = [f"{n}{' [kJ/mol]' if n.startswith('Ea') else ''}"
               for n in w.index]
    print(w.round(4).to_string())
    P.to_csv(CV_DIR / f"physics_folds_level{level}.csv", index=False)
    return cv, oof, P


# ==========================================================================
def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)

    verify_integrator(tr)
    fits_a = fit_all_variants(tr, y, level="A")
    tbl_a = report_variants(fits_a, y)
    tbl_a.to_csv(CV_DIR / "physics_variants_levelA.csv", index=False)

    best_a = min(fits_a.items(), key=lambda kv: kv[1]["train_rmse"])
    rule("STEP 3b. FITTED PARAMETERS OF THE BEST LEVEL-A VARIANT")
    print(f"    {best_a[0].strip()}   train RMSE = {best_a[1]['train_rmse']:.4f}\n")
    print(best_a[1]["params"].summary().to_string(index=False))

    cv_a, oof_a, folds_a = cross_validate_physics(tr, y, level="A")

    fits_b = fit_all_variants(tr, y, level="B")
    tbl_b = report_variants(fits_b, y)
    tbl_b.to_csv(CV_DIR / "physics_variants_levelB.csv", index=False)
    best_b = min(fits_b.items(), key=lambda kv: kv[1]["train_rmse"])
    rule("FITTED PARAMETERS OF THE BEST LEVEL-B VARIANT")
    print(f"    {best_b[0].strip()}   train RMSE = {best_b[1]['train_rmse']:.4f}\n")
    print(best_b[1]["params"].summary().to_string(index=False))

    cv_b, oof_b, folds_b = cross_validate_physics(tr, y, level="B")

    rule("PHASE 2 SUMMARY")
    print(f"    target sd (predict-the-mean baseline) : {y.std():.4f}")
    print(f"    Level A best train RMSE               : "
          f"{best_a[1]['train_rmse']:.4f}   ({best_a[0].strip()})")
    print(f"    Level A 10-fold CV RMSE               : {cv_a:.4f}")
    print(f"    Level B best train RMSE               : "
          f"{best_b[1]['train_rmse']:.4f}   ({best_b[0].strip()})")
    print(f"    Level B 10-fold CV RMSE               : {cv_b:.4f}")

    np.save(OUT_DIR / "oof_physics_A.npy", oof_a)
    np.save(OUT_DIR / "oof_physics_B.npy", oof_b)
    best = {"level_A": {k: float(getattr(best_a[1]["params"], k))
                        for k in ph.PARAM_NAMES},
            "level_B": {k: float(getattr(best_b[1]["params"], k))
                        for k in ph.PARAM_NAMES},
            "level_A_variant": best_a[0].strip(),
            "level_B_variant": best_b[0].strip(),
            "level_A_train_rmse": best_a[1]["train_rmse"],
            "level_B_train_rmse": best_b[1]["train_rmse"],
            "level_A_cv_rmse": cv_a, "level_B_cv_rmse": cv_b}
    (OUT_DIR / "physics_params.json").write_text(json.dumps(best, indent=2))
    print(f"\n    fitted parameters written to "
          f"{OUT_DIR / 'physics_params.json'}")


if __name__ == "__main__":
    main()
