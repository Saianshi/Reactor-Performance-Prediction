"""Phase 2d -- which physics parameters are actually identified by 150 rows?

Motivation. With the activation-energy ceiling raised to 500 kJ/mol, Ea2
stopped being pinned but started being UNSTABLE: across 10 folds it ranged
from 252 to 350 kJ/mol, and the folds that chose the largest Ea2 fitted their
training rows BEST while generalising WORST (fold 4: train 6.92, test 19.64).
That is the classic signature of a flat direction in the objective, not of
noise.

The physical reason is easy to state and worth stating in the pitch: once
k2*tau is large enough to destroy essentially all of the B, making it larger
changes nothing. Yield has already saturated at zero. So the 37 exact zeros --
a quarter of the dataset -- carry NO information about how far beyond the
cliff the operating point sits. Ea2 is therefore identified from below but
barely from above.

A profile likelihood shows this directly: fix one parameter on a grid,
re-optimise all the others, and plot the best achievable RMSE. A well
identified parameter gives a sharp minimum; a flat floor means the data cannot
distinguish the values along it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import physics as ph
from config import CV_DIR, FIG_DIR, SEED, TARGET, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")

FIXED_BASE = {"n_flow": 0.0}


def profile(df, y, name, grid, fixed_base=None, maxiter=25, popsize=10):
    """Best achievable RMSE with `name` held at each value in `grid`."""
    fixed_base = dict(fixed_base or FIXED_BASE)
    out = []
    for v in grid:
        fixed = dict(fixed_base)
        fixed[name] = float(v)
        p, r, _ = ph.fit_physics(df, y, level="B", fixed=fixed, seed=SEED,
                                 maxiter=maxiter, popsize=popsize,
                                 verbose=False)
        out.append({"value": v, "rmse": r,
                    **{k: getattr(p, k) for k in ph.PARAM_NAMES}})
        print(f"      {name} = {v:10.4g}   best RMSE = {r:8.4f}")
    return pd.DataFrame(out)


def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    y = tr[TARGET].to_numpy(float)

    print("=" * 80)
    print("PROFILE LIKELIHOOD -- how well does the data pin each parameter?")
    print("=" * 80)

    grids = {
        "Ea2": np.array([150, 200, 240, 280, 320, 380, 450]) * 1e3,
        "Ea1": np.array([30, 40, 45, 50, 55, 65, 80]) * 1e3,
        "gamma": np.array([-14, -10, -7, -4, -2, 0, 2]),
    }
    frames = {}
    for name, g in grids.items():
        print(f"\n  -- {name} --")
        frames[name] = profile(tr, y, name, g)
        frames[name].to_csv(CV_DIR / f"profile_{name}.csv", index=False)

    print("\n" + "=" * 80)
    print("IDENTIFIABILITY READ-OUT")
    print("=" * 80)
    for name, f in frames.items():
        best = f.rmse.min()
        # the set of values within 1% of the best achievable RMSE
        ok = f[f.rmse <= best * 1.01]
        lo, hi = ok.value.min(), ok.value.max()
        unit = "kJ/mol" if name.startswith("Ea") else "K.L/mol"
        scale = 1e-3 if name.startswith("Ea") else 1.0
        print(f"\n    {name}: best RMSE {best:.4f} at "
              f"{f.loc[f.rmse.idxmin(), 'value'] * scale:.3g} {unit}")
        print(f"      values within 1% of best: "
              f"[{lo * scale:.3g}, {hi * scale:.3g}] {unit}")
        span = (hi - lo) / max(abs(f.value).max(), 1e-12)
        print(f"      -> {'WELL IDENTIFIED' if span < 0.2 else 'WEAKLY IDENTIFIED (flat direction)'}")

    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (name, f) in zip(axes, frames.items()):
        x = f.value * (1e-3 if name.startswith("Ea") else 1.0)
        ax.plot(x, f.rmse, "o-", lw=2)
        best = f.rmse.min()
        ax.axhline(best * 1.01, color="r", ls="--",
                   label="1% above best achievable")
        ax.set_xlabel(f"{name} " + ("[kJ/mol]" if name.startswith("Ea")
                                    else "[K.L/mol]"))
        ax.set_ylabel("best achievable train RMSE")
        ax.set_title(f"profile likelihood: {name}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Which physics parameters does 150 rows actually identify?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_profile_likelihood.png", dpi=130,
                bbox_inches="tight")
    print(f"\n    figure -> {FIG_DIR / '04_profile_likelihood.png'}")


if __name__ == "__main__":
    main()
