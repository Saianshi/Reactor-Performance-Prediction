"""Phase 1 --- EDA done as a physicist.

Answers, with numbers rather than adjectives:
  Q1  Is the target bounded in [0, 100]? Where does it saturate?
  Q2  Is there ANY noise? (LOO residual scale of a flexible interpolator,
      fitted GP nugget, target quantisation, near-duplicate inputs)
  Q3  Does the test set sit inside the training convex hull, per-feature
      and jointly?
  Q4  Is the design a Latin hypercube, a grid, or plain uniform random?
  Q5  Does yield depend on CA0? For strictly first-order series kinetics the
      fractional yield of B is independent of CA0; any dependence is thermal
      (heat release scales with CA0) or the kinetics are not first order.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from config import FEATURES, FIG_DIR, TARGET, TEST_CSV, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==========================================================================
# Load
# ==========================================================================
def load():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    return tr, te


# ==========================================================================
# Q0 -- sanity
# ==========================================================================
def basic_checks(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    rule("Q0. STRUCTURAL SANITY")
    print(f"train shape {tr.shape},  test shape {te.shape}")
    print("\ndtypes:")
    print(pd.DataFrame({"train": tr.dtypes.astype(str),
                        "test": te.dtypes.astype(str)}).fillna("-"))
    print(f"\nmissing values  train={int(tr.isna().sum().sum())}  "
          f"test={int(te.isna().sum().sum())}")
    print(f"duplicate rows  train={int(tr.duplicated().sum())}  "
          f"test={int(te.duplicated().sum())}")
    print(f"duplicate feature-rows (ignoring target) train="
          f"{int(tr[FEATURES].duplicated().sum())}")
    # any train row repeated in test?
    merged = te.merge(tr[FEATURES].drop_duplicates(), on=FEATURES, how="inner")
    print(f"test rows exactly present in train: {len(merged)}")

    print("\nper-feature ranges:")
    desc = pd.DataFrame({
        "train_min": tr[FEATURES].min(), "train_max": tr[FEATURES].max(),
        "train_mean": tr[FEATURES].mean(), "train_std": tr[FEATURES].std(),
        "test_min": te[FEATURES].min(), "test_max": te[FEATURES].max(),
    })
    print(desc.round(3))

    # decimal places actually used -> tells us about the sampling grid
    print("\nnumber of decimal places used per column (max over rows):")
    for c in FEATURES + [TARGET]:
        s = tr[c] if c in tr else te[c]
        dp = s.astype(str).str.split(".").str[-1].str.len().max()
        print(f"  {c:24s} {dp}")


# ==========================================================================
# Q1 -- target support and saturation
# ==========================================================================
def target_checks(tr: pd.DataFrame) -> None:
    rule("Q1. TARGET SUPPORT AND SATURATION")
    y = tr[TARGET]
    print(y.describe().round(4).to_string())
    print(f"\nbounded in [0,100]? min={y.min():.4f}  max={y.max():.4f}  "
          f"-> {'YES' if (y.min() >= 0 and y.max() <= 100) else 'NO'}")
    print(f"exactly 0.0        : {(y == 0).sum():3d} rows "
          f"({100 * (y == 0).mean():.1f}%)")
    print(f"below 1.0          : {(y < 1).sum():3d} rows")
    print(f"below 5.0          : {(y < 5).sum():3d} rows")
    print(f"above 90.0         : {(y > 90).sum():3d} rows")
    print(f"above 95.0         : {(y > 95).sum():3d} rows")
    print(f"exactly 100.0      : {(y == 100).sum():3d} rows")
    print("\nhistogram (10 bins over observed range):")
    cnt, edges = np.histogram(y, bins=10)
    for c, lo, hi in zip(cnt, edges[:-1], edges[1:]):
        print(f"  [{lo:6.2f},{hi:6.2f})  {c:3d}  {'#' * c}")

    # what do the zero rows look like?
    zero = tr[tr[TARGET] == 0]
    if len(zero):
        print(f"\nthe {len(zero)} zero-yield rows -- feature ranges:")
        print(zero[FEATURES].describe().loc[["min", "mean", "max"]].round(2))
        print("\ncompare to the non-zero rows:")
        print(tr[tr[TARGET] > 0][FEATURES].describe()
              .loc[["min", "mean", "max"]].round(2))


# ==========================================================================
# Q2 -- is there any noise?
# ==========================================================================
def noise_checks(tr: pd.DataFrame) -> dict:
    """Four independent probes of the noise floor."""
    rule("Q2. IS THERE ANY NOISE? (deterministic simulator or plant data?)")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    from sklearn.preprocessing import StandardScaler

    X = tr[FEATURES].to_numpy(float)
    y = tr[TARGET].to_numpy(float)

    # -- probe 1: target quantisation ------------------------------------
    print("\n[1] target quantisation")
    dp = tr[TARGET].astype(str).str.split(".").str[-1].str.len().max()
    quant = 10.0 ** (-dp)
    print(f"    target reported to {dp} dp -> rounding step {quant:g}")
    print(f"    implied rounding noise sd = step/sqrt(12) = "
          f"{quant / np.sqrt(12):.6f} yield-%")
    print("    ANY noise floor we measure above this is model error, not "
          "measurement noise.")

    # -- probe 2: near-duplicate inputs with different y ------------------
    print("\n[2] near-duplicate inputs (replicates would expose run-to-run noise)")
    Xs = StandardScaler().fit_transform(X)
    d = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    print(f"    nearest-neighbour distance in standardised 5-D space:")
    print(f"      min={nn.min():.3f}  median={np.median(nn):.3f}  "
          f"max={nn.max():.3f}")
    print(f"    exact input replicates: {int((d < 1e-9).sum() // 2)} "
          "-> no repeated runs, so no direct noise estimate available")

    # -- probe 3: LOO of a flexible interpolator --------------------------
    print("\n[3] leave-one-out residuals of a flexible GP interpolator")
    print("    (Matern-5/2 + ARD, standardised inputs, tiny fixed nugget)")
    ker = (ConstantKernel(1.0, (1e-3, 1e6))
           * Matern(length_scale=np.ones(5), nu=2.5,
                    length_scale_bounds=(1e-2, 1e3)))
    sc = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=4, random_state=0)
    gp.fit(sc.transform(X), y)
    # exact LOO for a GP: mu_loo_i = y_i - [K^-1 y]_i / [K^-1]_ii
    K = gp.kernel_(sc.transform(X))
    K[np.diag_indices_from(K)] += 1e-8
    ym, ys = y.mean(), y.std()
    z = (y - ym) / ys
    Ki = np.linalg.inv(K)
    loo_res = (Ki @ z) / np.diag(Ki) * ys
    print(f"    LOO RMSE  = {np.sqrt((loo_res ** 2).mean()):.4f} yield-%")
    print(f"    LOO MAE   = {np.abs(loo_res).mean():.4f} yield-%")
    print(f"    LOO median|res| = {np.median(np.abs(loo_res)):.4f} yield-%")
    print(f"    target sd = {y.std():.4f} yield-%   "
          f"(so LOO R^2 ~ {1 - (loo_res ** 2).mean() / y.var():.4f})")

    # -- probe 4: let the GP estimate its own white-noise level -----------
    print("\n[4] fitted white-noise level (GP free to attribute variance to noise)")
    ker_w = (ConstantKernel(1.0, (1e-3, 1e6))
             * Matern(length_scale=np.ones(5), nu=2.5,
                      length_scale_bounds=(1e-2, 1e3))
             + WhiteKernel(1e-4, (1e-12, 1e1)))
    gpw = GaussianProcessRegressor(kernel=ker_w, normalize_y=True,
                                   n_restarts_optimizer=4, random_state=0)
    gpw.fit(sc.transform(X), y)
    wk = gpw.kernel_.k2.noise_level
    print(f"    fitted kernel: {gpw.kernel_}")
    print(f"    white-noise variance (normalised units) = {wk:.3e}")
    print(f"    -> noise sd = {np.sqrt(wk) * y.std():.4f} yield-%")
    print(f"    train residual RMSE = "
          f"{np.sqrt(((gpw.predict(sc.transform(X)) - y) ** 2).mean()):.4f}")

    print("\n    INTERPRETATION: the LOO error measures how well 150 points "
          "pin down a\n    smooth 5-D surface (sparse-coverage error), not "
          "observation noise. If the\n    fitted white-noise level collapses "
          "to its lower bound, the data carry no\n    detectable stochastic "
          "component -> interpolate, do not regularise hard.")
    return {"loo_rmse": float(np.sqrt((loo_res ** 2).mean())),
            "noise_sd": float(np.sqrt(wk) * y.std()),
            "loo_res": loo_res}


# ==========================================================================
# Q3 -- extrapolation
# ==========================================================================
def in_hull(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Is each row of `points` a convex combination of rows of `cloud`?

    Solved as an LP feasibility problem, which is exact and cheap here and
    does not need a Delaunay triangulation (intractable in 5-D).
    """
    n, d = cloud.shape
    A_eq = np.vstack([cloud.T, np.ones(n)])
    out = np.zeros(len(points), bool)
    for i, p in enumerate(points):
        b_eq = np.concatenate([p, [1.0]])
        r = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0, None)] * n, method="highs")
        out[i] = r.status == 0
    return out


def hull_checks(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    rule("Q3. DOES THE TEST SET SIT INSIDE THE TRAINING DOMAIN?")
    print("\n[per-feature box check]")
    rows = []
    for f in FEATURES:
        lo, hi = tr[f].min(), tr[f].max()
        below = int((te[f] < lo).sum())
        above = int((te[f] > hi).sum())
        # how far outside, as a fraction of the training range
        span = hi - lo
        worst = max(((lo - te[f].min()) / span) if below else 0.0,
                    ((te[f].max() - hi) / span) if above else 0.0)
        rows.append((f, lo, hi, te[f].min(), te[f].max(), below, above,
                     100 * worst))
    box = pd.DataFrame(rows, columns=["feature", "train_min", "train_max",
                                      "test_min", "test_max", "n_below",
                                      "n_above", "worst_excess_%range"])
    print(box.round(3).to_string(index=False))
    n_out = int((box.n_below + box.n_above).sum())
    print(f"\n-> {n_out} per-feature box violations in total")

    print("\n[joint convex-hull check, exact LP feasibility]")
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr[FEATURES])
    A = sc.transform(tr[FEATURES])
    B = sc.transform(te[FEATURES])
    inside = in_hull(B, A)
    print(f"    test rows inside the training convex hull: "
          f"{inside.sum()} / {len(B)}")
    if (~inside).any():
        print(f"    rows OUTSIDE (0-indexed): {np.where(~inside)[0].tolist()}")
        print(te.loc[~inside, FEATURES].round(2).to_string())

    try:
        h = ConvexHull(A)
        print(f"    training hull has {len(h.vertices)} vertices out of "
              f"{len(A)} points ({100 * len(h.vertices) / len(A):.0f}% are on "
              "the boundary)")
    except QhullError as e:
        print(f"    (hull vertex count unavailable: {e})")

    print("\n[local support: distance from each test point to training data]")
    d = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    nn1 = np.sort(d, axis=1)[:, 0]
    dtr = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dtr, np.inf)
    nn_tr = dtr.min(1)
    print(f"    test  -> nearest train : min={nn1.min():.3f} "
          f"median={np.median(nn1):.3f} max={nn1.max():.3f}")
    print(f"    train -> nearest train : min={nn_tr.min():.3f} "
          f"median={np.median(nn_tr):.3f} max={nn_tr.max():.3f}")
    flag = np.where(nn1 > np.percentile(nn_tr, 95))[0]
    print(f"    test rows further from training data than the 95th pct of "
          f"train-train NN distance: {flag.tolist()}")


# ==========================================================================
# Q4 -- design of experiments
# ==========================================================================
def design_checks(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    rule("Q4. WHAT IS THE DESIGN? (LHS / grid / uniform random)")
    both = pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True)

    print("\n[a] unique-value counts -- a grid would repeat levels")
    for f in FEATURES:
        print(f"    {f:24s} train {tr[f].nunique():3d}/{len(tr)}   "
              f"test {te[f].nunique():3d}/{len(te)}   "
              f"pooled {both[f].nunique():3d}/{len(both)}")

    print("\n[b] marginal uniformity (KS test vs Uniform[min,max]) on train")
    for f in FEATURES:
        s = tr[f]
        ks = stats.kstest((s - s.min()) / (s.max() - s.min()), "uniform")
        print(f"    {f:24s} KS={ks.statistic:.4f}  p={ks.pvalue:.3f}"
              f"   {'uniform' if ks.pvalue > 0.05 else 'NOT uniform'}")

    print("\n[c] LHS signature: with n strata of width range/n, an LHS puts "
          "exactly\n    one point in each stratum -> zero empty strata, max "
          "occupancy 1.")
    n = len(tr)
    for f in FEATURES:
        s = tr[f].to_numpy()
        lo, hi = s.min(), s.max()
        # use the pooled range so strata are not defined by the sample itself
        b = np.clip(((s - lo) / (hi - lo) * n).astype(int), 0, n - 1)
        occ = np.bincount(b, minlength=n)
        print(f"    {f:24s} empty strata {int((occ == 0).sum()):3d}/{n}   "
              f"max occupancy {occ.max()}")
    print("    (pure uniform random gives ~37% empty strata and max occupancy "
          "3-5;\n     a Latin hypercube gives 0 empty and max occupancy 1)")

    print("\n[d] input-input correlation (should be ~0 for LHS/random)")
    print(tr[FEATURES].corr().round(3).to_string())
    off = tr[FEATURES].corr().to_numpy()[np.triu_indices(5, 1)]
    print(f"    max |off-diagonal correlation| = {np.abs(off).max():.3f}")

    print("\n[e] are train and test drawn from the same distribution?")
    for f in FEATURES:
        ks = stats.ks_2samp(tr[f], te[f])
        print(f"    {f:24s} KS={ks.statistic:.4f}  p={ks.pvalue:.3f}"
              f"   {'same' if ks.pvalue > 0.05 else 'DIFFERENT'}")


# ==========================================================================
# Q5 -- concentration dependence
# ==========================================================================
def concentration_checks(tr: pd.DataFrame) -> None:
    rule("Q5. DOES YIELD DEPEND ON CA0? (first-order series kinetics says NO)")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_predict

    X = tr[FEATURES]
    y = tr[TARGET]
    ca = tr["concentration_mol_L"]

    print("\n[a] marginal association")
    print(f"    Pearson  r(CA0, yield) = {stats.pearsonr(ca, y)[0]:+.4f} "
          f"(p={stats.pearsonr(ca, y)[1]:.3f})")
    print(f"    Spearman r(CA0, yield) = {stats.spearmanr(ca, y)[0]:+.4f} "
          f"(p={stats.spearmanr(ca, y)[1]:.3f})")
    mi = mutual_info_regression(X, y, random_state=0)
    print("\n[b] mutual information with the target (nats)")
    for f, m in sorted(zip(FEATURES, mi), key=lambda t: -t[1]):
        print(f"    {f:24s} {m:.4f}")

    print("\n[c] permutation importance of a random forest "
          "(drop in R^2 when shuffled)")
    rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=0,
                                n_jobs=-1)
    for f, m, s in sorted(zip(FEATURES, pi.importances_mean,
                              pi.importances_std), key=lambda t: -t[1]):
        print(f"    {f:24s} {m:+.4f} +/- {s:.4f}")

    print("\n[d] the decisive test: does dropping CA0 hurt out-of-fold "
          "prediction?")
    from sklearn.model_selection import KFold
    kf = KFold(10, shuffle=True, random_state=0)
    for cols, label in [(FEATURES, "all 5 features"),
                        ([f for f in FEATURES if f != "concentration_mol_L"],
                         "WITHOUT CA0    ")]:
        p = cross_val_predict(
            RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1),
            tr[cols], y, cv=kf)
        print(f"    {label}  OOF RMSE = {np.sqrt(((p - y) ** 2).mean()):.4f}")

    print("\n[e] partial dependence of yield on CA0 (RF, all else averaged)")
    from sklearn.inspection import partial_dependence
    pd_res = partial_dependence(rf, X, ["concentration_mol_L"], grid_resolution=8)
    gv = pd_res["grid_values"][0]
    av = pd_res["average"][0]
    for g, a in zip(gv, av):
        print(f"    CA0={g:5.2f} mol/L -> yield {a:6.2f} %")
    print(f"    total swing across the CA0 range: "
          f"{av.max() - av.min():.2f} yield-%")

    print("\n[f] residual test: fit yield on the OTHER four features, then ask "
          "whether\n    the residual still correlates with CA0")
    other = [f for f in FEATURES if f != "concentration_mol_L"]
    p = cross_val_predict(
        RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1),
        tr[other], y, cv=kf)
    res = y - p
    r, pv = stats.pearsonr(ca, res)
    print(f"    Pearson r(CA0, residual) = {r:+.4f}  (p={pv:.4f})")
    print("    A non-zero value here is the thermal fingerprint: heat release "
          "scales\n    with CA0, so a more concentrated feed runs hotter and "
          "sits at a different\n    point on the k2/k1 selectivity curve.")


# ==========================================================================
# Plots
# ==========================================================================
def make_plots(tr: pd.DataFrame, te: pd.DataFrame, loo_res: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = tr["length_m"] / tr["flow_rate_L_min"]
    tau_te = te["length_m"] / te["flow_rate_L_min"]

    # -- yield vs each feature + tau ------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, f in zip(axes.ravel(), FEATURES):
        sc = ax.scatter(tr[f], tr[TARGET], c=tr["jacket_temperature_K"],
                        cmap="coolwarm", s=32, edgecolor="k", linewidth=.3)
        ax.set_xlabel(f)
        ax.set_ylabel(TARGET)
        ax.set_title(f"yield vs {f}")
        ax.grid(alpha=.3)
    ax = axes.ravel()[5]
    ax.scatter(tau, tr[TARGET], c=tr["jacket_temperature_K"], cmap="coolwarm",
               s=32, edgecolor="k", linewidth=.3)
    ax.set_xscale("log")
    ax.set_xlabel("tau_proxy = L / Q   [m.min/L]")
    ax.set_ylabel(TARGET)
    ax.set_title("yield vs residence-time proxy (log x)\ncolour = jacket T")
    ax.grid(alpha=.3)

    ax = axes.ravel()[6]
    ax.scatter(np.log(tau), tr["jacket_temperature_K"], c=tr[TARGET],
               cmap="viridis", s=40, edgecolor="k", linewidth=.3)
    ax.set_xlabel("log tau_proxy")
    ax.set_ylabel("jacket temperature [K]")
    ax.set_title("the operating map: colour = yield")
    ax.grid(alpha=.3)

    ax = axes.ravel()[7]
    ax.hist(tr[TARGET], bins=30, color="steelblue", edgecolor="k")
    ax.set_xlabel(TARGET)
    ax.set_title("target distribution")
    ax.grid(alpha=.3)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=.6,
                 label="jacket temperature [K]")
    fig.suptitle("Phase 1 EDA -- yield against every input", fontsize=15)
    fig.savefig(FIG_DIR / "01_yield_vs_features.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # -- train vs test coverage ------------------------------------------
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.6))
    for ax, f in zip(axes, FEATURES):
        ax.hist(tr[f], bins=15, alpha=.6, label="train", density=True)
        ax.hist(te[f], bins=15, alpha=.6, label="test", density=True)
        ax.set_title(f, fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend()
    axes[5].hist(tau, bins=15, alpha=.6, label="train", density=True)
    axes[5].hist(tau_te, bins=15, alpha=.6, label="test", density=True)
    axes[5].set_title("tau_proxy = L/Q", fontsize=9)
    axes[5].grid(alpha=.3)
    fig.suptitle("Train vs test input coverage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_train_test_coverage.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # -- LOO residuals ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(tr[TARGET], tr[TARGET] - loo_res, s=25, edgecolor="k",
                    linewidth=.3)
    lims = [tr[TARGET].min() - 3, tr[TARGET].max() + 3]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("actual")
    axes[0].set_ylabel("LOO prediction")
    axes[0].set_title("GP leave-one-out")
    axes[1].scatter(tr[TARGET], loo_res, s=25, edgecolor="k", linewidth=.3)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("actual")
    axes[1].set_ylabel("LOO residual")
    axes[1].set_title("residual vs level")
    axes[2].hist(loo_res, bins=30, color="steelblue", edgecolor="k")
    axes[2].set_title("LOO residual distribution")
    for a in axes:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_loo_residuals.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigures written to {FIG_DIR}")


# ==========================================================================
def main() -> None:
    set_seed()
    tr, te = load()
    basic_checks(tr, te)
    target_checks(tr)
    nz = noise_checks(tr)
    hull_checks(tr, te)
    design_checks(tr, te)
    concentration_checks(tr)
    make_plots(tr, te, nz["loo_res"])


if __name__ == "__main__":
    main()
