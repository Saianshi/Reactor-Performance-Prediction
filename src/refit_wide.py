"""Phase 2c -- refit Level B with the activation-energy ceiling raised, and
check every parameter for bound pinning.

Also settles the n_flow contradiction. Level A said n_flow = +0.06 +/- 0.06
(indistinguishable from zero, matching the GP's verdict that Q carries no
information beyond tau). Level B said n_flow = -0.214 +/- 0.027, which is 8
sigma from zero AND almost exactly the turbulent-flow prediction of -0.2.
Both cannot be right, and the difference matters for the pitch, so the
comparison is run here with Ea2 free to move.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import physics as ph
from config import CV_DIR, OUT_DIR, SEED, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

VARIANTS = {
    "M0 beta const, gamma=0   ": {"n_flow": 0.0, "gamma": 0.0},
    "M1 beta const, gamma free": {"n_flow": 0.0},
    "M2 beta(Q), gamma=0      ": {"gamma": 0.0},
    "M3 beta(Q), gamma free   ": {},
}


def rule(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


def fit_one(df, y, fixed, maxiter=90, popsize=16):
    # fit_steps=60 for the SEARCH (integration error ~0.016 RMSE, three orders
    # below the signal) and final_steps=400 for the reported value. Spending
    # the saved time on more DE iterations instead buys far more than a finer
    # integration grid would: the earlier fold-to-fold scatter in Ea2 came from
    # an under-converged search, not from integration error.
    return ph.fit_physics(df, y, level="B", fixed=fixed, seed=SEED,
                          maxiter=maxiter, popsize=popsize, verbose=False,
                          fit_steps=60, final_steps=150, polish_steps=120)


def cv_one(df, y, fixed, n_splits=10, verbose=True):
    kf = KFold(n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    rows = []
    for i, (tri, tei) in enumerate(kf.split(df)):
        p, r, _ = fit_one(df.iloc[tri], y[tri], fixed)
        oof[tei] = ph.level_b(p, df.iloc[tei], n_steps=400)
        d = {k: getattr(p, k) for k in ph.PARAM_NAMES}
        d["fold"] = i
        d["train_rmse"] = r
        d["test_rmse"] = ph.rmse(oof[tei], y[tei])
        rows.append(d)
        if verbose:
            print(f"          fold {i:2d}  train {r:7.3f}  test "
                  f"{d['test_rmse']:8.3f}   Ea1={p.Ea1/1e3:6.1f} "
                  f"Ea2={p.Ea2/1e3:6.1f}  gamma={p.gamma:+6.2f}")
    return ph.rmse(oof, y), oof, pd.DataFrame(rows)


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)

    rule("LEVEL B REFIT WITH Ea BOUNDS WIDENED TO 10-500 kJ/mol")
    print(f"    previous ceiling was 250 kJ/mol and Ea2 pinned there in all "
          f"10 folds\n    (sd exactly 0.0) -- so that value was a bound, not "
          f"an estimate.\n")

    out = {}
    for name, fixed in VARIANTS.items():
        t0 = time.time()
        p, r, _ = fit_one(tr, y, fixed)
        cv, oof, P = cv_one(tr, y, fixed)
        k = 7 - len(fixed)
        aic = len(y) * np.log(r ** 2) + 2 * k
        out[name] = {"p": p, "train": r, "cv": cv, "k": k, "aic": aic,
                     "folds": P, "oof": oof}
        print(f"    {name}  k={k}  train={r:7.4f}  CV={cv:7.4f}  AIC={aic:7.1f}"
              f"   ({time.time() - t0:.0f}s)")
        print(f"        Ea1={p.Ea1/1e3:7.2f}  Ea2={p.Ea2/1e3:7.2f} kJ/mol   "
              f"beta0={p.beta0:7.3f}  n_flow={p.n_flow:+.4f}  "
              f"gamma={p.gamma:+7.3f}")
        pin = ph.check_pinning(p, fixed=fixed)
        bad = pin[pin.status.str.startswith("PINNED")]
        if len(bad):
            print("        !! STILL PINNED: "
                  + ", ".join(f"{r_.parameter} at {r_.value:.4g}"
                              for _, r_ in bad.iterrows()))
        else:
            print("        all free parameters interior to their bounds  OK")
        P.to_csv(CV_DIR / f"wide_folds_{name.split()[0]}.csv", index=False)

    # ------------------------------------------------------------------
    rule("VARIANT TABLE")
    t = pd.DataFrame([{
        "variant": n.strip(), "k": v["k"], "train": v["train"], "CV": v["cv"],
        "AIC": v["aic"], "Ea1_kJ": v["p"].Ea1 / 1e3, "Ea2_kJ": v["p"].Ea2 / 1e3,
        "lnA1": v["p"].lnA1, "lnA2": v["p"].lnA2, "beta0": v["p"].beta0,
        "n_flow": v["p"].n_flow, "gamma": v["p"].gamma}
        for n, v in out.items()])
    print(t.round(4).to_string(index=False))
    t.to_csv(CV_DIR / "levelB_wide_variants.csv", index=False)

    # ------------------------------------------------------------------
    rule("THE n_flow QUESTION, SETTLED")
    pairs = [("M2 beta(Q), gamma=0      ", "M0 beta const, gamma=0   "),
             ("M3 beta(Q), gamma free   ", "M1 beta const, gamma free")]
    for flow_name, const_name in pairs:
        P = out[flow_name]["folds"]
        m, s = P.n_flow.mean(), P.n_flow.std()
        print(f"    {flow_name.strip():26s} n_flow = {m:+.4f} +/- {s:.4f}  "
              f"({abs(m / max(s, 1e-9)):.1f} sd from zero)")
        print(f"        turbulent internal flow (h ~ Re^0.8) predicts -0.200")
        print(f"        CV with beta(Q) = {out[flow_name]['cv']:7.4f}    "
              f"CV with beta const = {out[const_name]['cv']:7.4f}    "
              f"gain = {out[const_name]['cv'] - out[flow_name]['cv']:+.4f}")

    rule("PARAMETER STABILITY OF THE BEST VARIANT BY CV")
    best_name = min(out, key=lambda n: out[n]["cv"])
    P = out[best_name]["folds"]
    print(f"    {best_name.strip()}   CV = {out[best_name]['cv']:.4f}\n")
    w = P[ph.PARAM_NAMES].agg(["mean", "std", "min", "max"]).T
    w["cv_%"] = 100 * w["std"] / w["mean"].abs().replace(0, np.nan)
    for c in ["Ea1", "Ea2"]:
        w.loc[c, ["mean", "std", "min", "max"]] /= 1000
    print(w.round(4).to_string())
    print(f"\n    fold test RMSE: min={P.test_rmse.min():.3f}  "
          f"median={P.test_rmse.median():.3f}  max={P.test_rmse.max():.3f}")

    print("\n    PHYSICAL PLAUSIBILITY CHECK")
    p = out[best_name]["p"]
    print(f"      Ea1 = {p.Ea1/1e3:.1f} kJ/mol   "
          f"{'plausible (typical liquid-phase reaction 40-150)' if 20e3 < p.Ea1 < 200e3 else 'UNUSUAL'}")
    print(f"      Ea2 = {p.Ea2/1e3:.1f} kJ/mol   "
          f"{'plausible' if 20e3 < p.Ea2 < 250e3 else 'VERY HIGH -- report honestly, do not hide'}")
    print(f"      Ea2 - Ea1 = {(p.Ea2 - p.Ea1)/1e3:.1f} kJ/mol "
          f"{'(>0 OK: selectivity degrades with T)' if p.Ea2 > p.Ea1 else '(<0 REJECT)'}")
    assert p.Ea2 > p.Ea1, "Ea2 < Ea1 contradicts the observed high-T collapse"

    best = {"variant": best_name.strip(),
            "params": {k: float(getattr(p, k)) for k in ph.PARAM_NAMES},
            "train_rmse": out[best_name]["train"], "cv_rmse": out[best_name]["cv"],
            "all": {n.strip(): {"train": v["train"], "cv": v["cv"], "k": v["k"]}
                    for n, v in out.items()}}
    (OUT_DIR / "physics_params_wide.json").write_text(json.dumps(best, indent=2))
    np.save(OUT_DIR / "oof_physics_best.npy", out[best_name]["oof"])
    print(f"\n    written to {OUT_DIR / 'physics_params_wide.json'}")


if __name__ == "__main__":
    main()
