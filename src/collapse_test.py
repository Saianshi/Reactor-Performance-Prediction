"""Phase 2a -- THE DIMENSION COLLAPSE TEST.

Hypothesis. For strictly first-order series kinetics the fractional yield of B
depends only on the dimensionless groups Da1 = k1*tau and Da2 = k2*tau. The
thermal model enters through beta*L/Q = beta*tau. So L and Q should act ONLY
through tau = L/Q, and the response surface is f(tau, T0, Tj) -- a 3-D function
sampled 150 times, not a 5-D one.

Three ways of testing it, because a single GP LOO number is not trustworthy:
  T1  LOO RMSE across coordinate systems, repeated over seeds/restarts so we
      can see the optimiser scatter and not over-read a 1-point difference.
  T2  The direct residual test: fit on (log tau, 1/T0, 1/Tj) alone, then ask
      whether the residual still knows about log Q. If tau is sufficient, it
      cannot.
  T3  Matched-pair test: rows with near-identical (tau, T0, Tj) but very
      different Q. Under collapse their yields must agree.

Also settles the zero floor: are all 37 exact zeros at high thermal severity
(a true asymptotic decay), or does one sit at low severity (a clipping
artefact, which would change everything)?
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from config import CV_DIR, FEATURES, R_GAS, TARGET, TEST_CSV, TRAIN_CSV, set_seed

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def rule(t):
    print("\n" + "=" * 82)
    print(t)
    print("=" * 82)


# --------------------------------------------------------------------------
def gp_loo(X, y, seed=0, restarts=15, return_pred=False):
    """Exact LOO of an interpolating Matern-5/2 ARD GP (clipped to [0,100])."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    ker = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
        length_scale=np.ones(X.shape[1]), nu=2.5,
        length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=restarts,
                                  random_state=seed)
    gp.fit(Xs, y)
    K = gp.kernel_(Xs)
    K[np.diag_indices_from(K)] += 1e-8
    m, s = y.mean(), y.std()
    Ki = np.linalg.inv(K)
    pred = np.clip(y - (Ki @ ((y - m) / s)) / np.diag(Ki) * s, 0, 100)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    if return_pred:
        return rmse, pred, gp.kernel_
    return rmse


def build_coords(df):
    Q = df.flow_rate_L_min.to_numpy(float)
    L = df.length_m.to_numpy(float)
    T0 = df.inlet_temperature_K.to_numpy(float)
    Tj = df.jacket_temperature_K.to_numpy(float)
    CA = df.concentration_mol_L.to_numpy(float)
    tau = L / Q
    return {
        "(c) raw 5 inputs                      ": df[FEATURES].to_numpy(float),
        "(c) logtau,logQ,logL,T0,Tj            ": np.c_[np.log(tau), np.log(Q),
                                                        np.log(L), T0, Tj],
        "(c) logtau,logQ,1/T0,1/Tj,CA0         ": np.c_[np.log(tau), np.log(Q),
                                                        1 / T0, 1 / Tj, CA],
        "(a) logtau,1/T0,1/Tj          [3-D]   ": np.c_[np.log(tau), 1 / T0,
                                                        1 / Tj],
        "(a') logtau,T0,Tj             [3-D]   ": np.c_[np.log(tau), T0, Tj],
        "(b) logtau,1/T0,1/Tj,logQ     [4-D]   ": np.c_[np.log(tau), 1 / T0,
                                                        1 / Tj, np.log(Q)],
        "(b') logtau,1/T0,1/Tj,CA0     [4-D]   ": np.c_[np.log(tau), 1 / T0,
                                                        1 / Tj, CA],
        "(d) logtau,1/T0,1/Tj,logQ,CA0 [5-D]   ": np.c_[np.log(tau), 1 / T0,
                                                        1 / Tj, np.log(Q), CA],
    }


# --------------------------------------------------------------------------
def t1_loo_table(tr):
    rule("T1. LOO RMSE BY COORDINATE SYSTEM (5 seeds x 15 restarts each)")
    print("    Repeated because GP marginal-likelihood optimisation has local")
    print("    optima; a single fit can move by ~1 RMSE point on its own.\n")
    y = tr[TARGET].to_numpy(float)
    coords = build_coords(tr)
    seeds = [0, 1, 2, 3, 4]
    rows = []
    print(f"    {'coordinates':40s}{'d':>3}{'mean':>9}{'sd':>8}"
          f"{'min':>9}{'max':>9}")
    print("    " + "-" * 78)
    for name, X in coords.items():
        vals = [gp_loo(X, y, seed=s) for s in seeds]
        rows.append({"coords": name.strip(), "d": X.shape[1],
                     "mean": np.mean(vals), "sd": np.std(vals),
                     "min": np.min(vals), "max": np.max(vals)})
        print(f"    {name:40s}{X.shape[1]:>3}{np.mean(vals):9.3f}"
              f"{np.std(vals):8.3f}{np.min(vals):9.3f}{np.max(vals):9.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(CV_DIR / "collapse_loo_by_coords.csv", index=False)
    print(f"\n    baseline (predict the mean) = {y.std():.3f}")
    return df


# --------------------------------------------------------------------------
def t2_residual_test(tr):
    rule("T2. THE DIRECT TEST -- does the 3-D residual still know about Q?")
    print("    Fit yield on (log tau, 1/T0, 1/Tj) ONLY. If tau = L/Q is a")
    print("    sufficient statistic for the pair (L, Q), then the leftover")
    print("    residual must be independent of Q and of L separately.\n")
    y = tr[TARGET].to_numpy(float)
    Q = tr.flow_rate_L_min.to_numpy(float)
    L = tr.length_m.to_numpy(float)
    T0 = tr.inlet_temperature_K.to_numpy(float)
    Tj = tr.jacket_temperature_K.to_numpy(float)
    CA = tr.concentration_mol_L.to_numpy(float)
    tau = L / Q

    X3 = np.c_[np.log(tau), 1 / T0, 1 / Tj]
    rmse, pred, ker = gp_loo(X3, y, seed=0, restarts=15, return_pred=True)
    res = y - pred
    print(f"    3-D GP LOO RMSE = {rmse:.3f}")
    print(f"    fitted kernel   : {ker}\n")
    print(f"    {'variable':22s}{'Pearson r':>12}{'p':>10}"
          f"{'Spearman r':>13}{'p':>10}")
    print("    " + "-" * 67)
    for nm, v in [("log Q", np.log(Q)), ("log L", np.log(L)),
                  ("CA0", CA), ("log tau", np.log(tau))]:
        pr, pp = stats.pearsonr(v, res)
        sr, sp = stats.spearmanr(v, res)
        print(f"    {nm:22s}{pr:+12.4f}{pp:10.4f}{sr:+13.4f}{sp:10.4f}")

    print("\n    Same test, but on |residual| (a Q-dependence could be in the")
    print("    scatter rather than the mean):")
    for nm, v in [("log Q", np.log(Q)), ("log L", np.log(L)), ("CA0", CA)]:
        pr, pp = stats.pearsonr(v, np.abs(res))
        print(f"    {nm:22s}{pr:+12.4f}{pp:10.4f}")

    print("\n    Can a second GP predict the residual FROM log Q and CA0 alone?")
    print("    (if yes, there is real left-over structure; if no, tau suffices)")
    for nm, Xr in [("log Q only     ", np.log(Q)[:, None]),
                   ("CA0 only       ", CA[:, None]),
                   ("log Q + CA0    ", np.c_[np.log(Q), CA]),
                   ("log Q+CA0+logL ", np.c_[np.log(Q), CA, np.log(L)])]:
        r2 = gp_loo(Xr, res, seed=0, restarts=10)
        print(f"      {nm} LOO RMSE on residual = {r2:7.3f}   "
              f"(residual sd = {res.std():.3f})  "
              f"{'-> structure found' if r2 < res.std() * 0.95 else '-> nothing'}")
    return res


# --------------------------------------------------------------------------
def t3_matched_pairs(tr):
    rule("T3. MATCHED PAIRS -- same (tau, T0, Tj), very different Q")
    print("    Under collapse these rows are the SAME physical operating point")
    print("    and must have the same yield, however different Q and L are.\n")
    y = tr[TARGET].to_numpy(float)
    Q = tr.flow_rate_L_min.to_numpy(float)
    L = tr.length_m.to_numpy(float)
    T0 = tr.inlet_temperature_K.to_numpy(float)
    Tj = tr.jacket_temperature_K.to_numpy(float)
    tau = L / Q

    Z = np.c_[np.log(tau), T0, Tj]
    Z = (Z - Z.mean(0)) / Z.std(0)
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    iu = np.triu_indices(len(Z), 1)
    dd = d[iu]
    qratio = np.abs(np.log(Q)[iu[0]] - np.log(Q)[iu[1]])
    # closest in the 3-D collapsed space AND far apart in Q
    score = dd + 0.0 * qratio
    cand = np.where((dd < 0.35) & (qratio > 0.7))[0]
    cand = cand[np.argsort(dd[cand])][:12]
    if len(cand) == 0:
        print("    no sufficiently matched pairs at this tolerance")
        return
    print(f"    {'d3':>6}{'Q_i':>8}{'Q_j':>8}{'Qratio':>8}{'L_i':>7}{'L_j':>7}"
          f"{'tau_i':>8}{'tau_j':>8}{'y_i':>9}{'y_j':>9}{'|dy|':>8}")
    print("    " + "-" * 90)
    dys = []
    for k in cand:
        i, j = iu[0][k], iu[1][k]
        dys.append(abs(y[i] - y[j]))
        print(f"    {d[i, j]:6.3f}{Q[i]:8.2f}{Q[j]:8.2f}"
              f"{max(Q[i], Q[j]) / min(Q[i], Q[j]):8.2f}{L[i]:7.2f}{L[j]:7.2f}"
              f"{tau[i]:8.3f}{tau[j]:8.3f}{y[i]:9.3f}{y[j]:9.3f}"
              f"{abs(y[i] - y[j]):8.3f}")
    print(f"\n    n={len(cand)} matched pairs, Q differing by up to "
          f"{max(max(Q[iu[0][k]], Q[iu[1][k]]) / min(Q[iu[0][k]], Q[iu[1][k]]) for k in cand):.1f}x")
    print(f"    mean |dy| = {np.mean(dys):.3f} yield-%   "
          f"median |dy| = {np.median(dys):.3f}")
    print(f"    for scale, target sd = {y.std():.2f} and the 3-D GP LOO RMSE "
          "is ~16")


# --------------------------------------------------------------------------
def t4_zero_floor(tr):
    rule("T4. ARE THE 37 EXACT ZEROS A TRUE ASYMPTOTIC DECAY OR A CLIP?")
    print("    A true decay means every zero sits at high thermal severity")
    print("    Da2 = k2*tau >> 1. Even ONE zero at low severity would mean the")
    print("    organisers clipped negatives, which changes the whole approach.\n")
    y = tr[TARGET].to_numpy(float)
    Q = tr.flow_rate_L_min.to_numpy(float)
    L = tr.length_m.to_numpy(float)
    T0 = tr.inlet_temperature_K.to_numpy(float)
    Tj = tr.jacket_temperature_K.to_numpy(float)
    tau = L / Q
    zero = y == 0

    # Severity index S = tau * exp(-Ea/(R*Tref)). Scan Ea and the temperature
    # the fluid actually sees, and ask for PERFECT separation of the zeros.
    print("    Scanning a severity index S = tau*exp(-Ea/(R*T)) for the Ea and")
    print("    reference temperature that best separate zeros from non-zeros:\n")
    best = None
    for wj in [1.0, 0.8, 0.6, 0.5]:                 # weight on Tj vs T0
        Tref = wj * Tj + (1 - wj) * T0
        for Ea in np.arange(20e3, 260e3, 2.5e3):
            S = np.log(tau) - Ea / (R_GAS * Tref)
            # separation quality: how many non-zeros sit above the lowest zero
            thr = S[zero].min()
            n_viol = int((S[~zero] > thr).sum())
            auc = stats.mannwhitneyu(S[zero], S[~zero]).statistic / (
                zero.sum() * (~zero).sum())
            if best is None or (n_viol, -auc) < (best[0], -best[1]):
                best = (n_viol, auc, Ea, wj, S.copy(), thr)
    n_viol, auc, Ea, wj, S, thr = best
    print(f"    best separating index: Ea = {Ea / 1000:.0f} kJ/mol, "
          f"T_ref = {wj:.1f}*Tj + {1 - wj:.1f}*T0")
    print(f"    AUC (zeros vs non-zeros) = {auc:.4f}   "
          f"{'PERFECT separation' if auc == 1.0 else ''}")
    print(f"    non-zero rows sitting above the least-severe zero: {n_viol}")

    order = np.argsort(-S)
    print(f"\n    the 45 most severe rows, ranked by S (zeros should be a "
          "contiguous block at the top):")
    print(f"    {'rank':>5}{'S':>9}{'tau':>8}{'Tj':>8}{'T0':>8}{'yield':>10}")
    print("    " + "-" * 48)
    for r, i in enumerate(order[:45]):
        mark = "  <-- ZERO" if zero[i] else ""
        print(f"    {r:5d}{S[i]:9.3f}{tau[i]:8.3f}{Tj[i]:8.1f}{T0[i]:8.1f}"
              f"{y[i]:10.3f}{mark}")

    print(f"\n    least severe ZERO      : S = {S[zero].min():.3f}")
    print(f"    most severe NON-zero   : S = {S[~zero].max():.3f}  "
          f"(yield = {y[~zero][np.argmax(S[~zero])]:.3f})")
    if S[~zero].max() < S[zero].min():
        print("    -> the zeros form a clean contiguous high-severity block.")
        print("       CONSISTENT WITH TRUE ASYMPTOTIC DECAY, not a clip.")
    else:
        n_overlap = int((S[~zero] > S[zero].min()).sum())
        print(f"    -> {n_overlap} non-zero rows are more severe than the "
              "least-severe zero.")
        print("       Inspect them: small non-zero yields in the overlap are")
        print("       fine (decay is continuous); a LARGE yield would not be.")
        ov = np.where((~zero) & (S > S[zero].min()))[0]
        print(f"\n    the {len(ov)} overlapping non-zero rows:")
        print(f"    {'S':>9}{'tau':>8}{'Tj':>8}{'T0':>8}{'yield':>10}")
        for i in ov[np.argsort(-S[ov])]:
            print(f"    {S[i]:9.3f}{tau[i]:8.3f}{Tj[i]:8.1f}{T0[i]:8.1f}"
                  f"{y[i]:10.3f}")
        print(f"\n    max yield among overlapping rows = {y[ov].max():.3f}")
        print("    (if this is small, the decay is smooth and continuous ->")
        print("     still a true decay, just a soft boundary in a crude index)")

    print(f"\n    smallest NON-zero yields in the data "
          "(a true decay approaches 0 continuously):")
    nz = np.sort(y[y > 0])[:15]
    print("    " + "  ".join(f"{v:.3f}" for v in nz))


# --------------------------------------------------------------------------
def main():
    set_seed()
    tr = pd.read_csv(TRAIN_CSV)
    t1_loo_table(tr)
    t2_residual_test(tr)
    t3_matched_pairs(tr)
    t4_zero_floor(tr)


if __name__ == "__main__":
    main()
