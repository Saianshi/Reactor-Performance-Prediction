"""Phase 3 -- feature engineering.

Every feature here is either a raw input, a dimensionless group with a
physical meaning, or an output of the fitted reactor model. Nothing is a
blind polynomial expansion: with 150 rows, each feature has to earn its place.

A note on log Q and log L. The brief asks for them, and they are built here,
but Phase 2a established that L and Q act ONLY through tau = L/Q -- supplying
them separately cost +3.78 RMSE, and the GP's ARD kernel pinned log Q's
length-scale at its upper bound. They are therefore included in the FULL set
and expected to be pruned; the pruning is done by cross-validated selection,
not by assertion, so the data gets the final say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import physics as ph
from config import FEATURES, R_GAS

# fixed heat-transfer groups used as a robustness hedge, so the feature set
# does not depend on a single fitted beta
BETA_HEDGE = [1.0, 10.0, 100.0]


def build_features(df: pd.DataFrame, p: ph.PhysParams,
                   include_raw_LQ: bool = True) -> pd.DataFrame:
    """Full engineered feature matrix from the five raw inputs."""
    Q = df["flow_rate_L_min"].to_numpy(float)
    CA0 = df["concentration_mol_L"].to_numpy(float)
    T0 = df["inlet_temperature_K"].to_numpy(float)
    L = df["length_m"].to_numpy(float)
    Tj = df["jacket_temperature_K"].to_numpy(float)

    X = pd.DataFrame(index=df.index)

    # -- raw ------------------------------------------------------------
    for f in FEATURES:
        X[f] = df[f].to_numpy(float)

    # -- residence time -------------------------------------------------
    tau = L / Q
    X["tau"] = tau
    X["log_tau"] = np.log(tau)
    if include_raw_LQ:
        X["Q_over_L"] = Q / L
        X["log_Q"] = np.log(Q)
        X["log_L"] = np.log(L)

    # -- temperature differences ----------------------------------------
    X["delta_T"] = Tj - T0
    X["Tj_over_T0"] = Tj / T0
    X["inv_T0"] = 1.0 / T0
    X["inv_Tj"] = 1.0 / Tj

    # -- thermal model: fitted beta, plus fixed hedges -------------------
    beta_fit = ph.beta_of_Q(p.log_beta0, p.n_flow, Q)
    Teff_fit = ph.t_eff(T0, Tj, beta_fit, tau)
    X["T_eff"] = Teff_fit
    X["inv_T_eff"] = 1.0 / Teff_fit
    X["b_number"] = beta_fit * tau          # dimensionless heating length
    for b in BETA_HEDGE:
        X[f"T_eff_b{b:g}"] = ph.t_eff(T0, Tj, b, tau)

    # -- kinetics at the effective temperature --------------------------
    A = ph.level_a(p, df, full=True)
    k1, k2 = A["k1"], A["k2"]
    X["log_k1"] = np.log(np.maximum(k1, 1e-300))
    X["log_k2"] = np.log(np.maximum(k2, 1e-300))
    X["log_k_ratio"] = X["log_k2"] - X["log_k1"]      # selectivity driver

    # -- Damkohler numbers ----------------------------------------------
    X["log_Da1"] = X["log_k1"] + np.log(tau)
    X["log_Da2"] = X["log_k2"] + np.log(tau)
    X["Da1"] = np.clip(k1 * tau, 0, 1e6)
    X["Da2"] = np.clip(k2 * tau, 0, 1e6)

    # -- the optimum ----------------------------------------------------
    # tau/tau_opt < 1 : under-converted, the reactor is too short/too fast
    # tau/tau_opt > 1 : over-reacting, B is being burned to C
    X["tau_over_tau_opt"] = tau / np.maximum(A["tau_opt"], 1e-12)
    X["log_tau_over_tau_opt"] = np.log(np.maximum(X["tau_over_tau_opt"], 1e-12))
    X["Y_max"] = A["Y_max"]
    X["conversion_X"] = A["conversion_X"]

    # -- the physics predictions themselves ------------------------------
    X["Y_phys_A"] = A["Y_phys_A"]
    X["Y_phys_B"] = ph.level_b(p, df, n_steps=400)

    # -- heat release proxy ----------------------------------------------
    # CA0 enters the true model only through gamma*CA0, i.e. only as a
    # temperature shift weighted by how much A has actually reacted.
    X["CA0_Da1"] = CA0 * X["Da1"]
    X["CA0_X"] = CA0 * A["conversion_X"]
    X["T_rise"] = p.gamma * CA0 * A["conversion_X"]

    # A non-finite feature must never reach a regressor. This should not
    # trigger for a converged physics fit, but during cross-validation the
    # physics is refitted on 90% of the rows and an occasional fold can land
    # somewhere extreme, so fail loudly here rather than let sklearn raise a
    # confusing error three call frames away.
    bad = X.columns[~np.isfinite(X.to_numpy(float)).all(axis=0)]
    if len(bad):
        raise FloatingPointError(
            f"non-finite engineered features: {list(bad)} "
            f"(physics params: {p})")
    return X


PAIR_PRODUCTS = [
    ("log_tau", "delta_T"),
    ("log_tau", "jacket_temperature_K"),
    ("inlet_temperature_K", "jacket_temperature_K"),
    ("concentration_mol_L", "log_tau"),
]


def add_pair_products(X: pd.DataFrame, ref: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """A handful of standardised pairwise products, standardised using `ref`.

    `ref` must be the TRAINING matrix when transforming validation or test
    data, so that no test statistic leaks into the standardisation.
    """
    src = X if ref is None else ref
    out = X.copy()
    for a, b in PAIR_PRODUCTS:
        za = (X[a] - src[a].mean()) / (src[a].std() + 1e-12)
        zb = (X[b] - src[b].mean()) / (src[b].std() + 1e-12)
        out[f"x_{a}__{b}"] = za * zb
    return out


# --------------------------------------------------------------------------
# Curated subsets, smallest first. Selection between them is done by
# cross-validation in Phase 4, never by looking at the test set.
# --------------------------------------------------------------------------
FEATURE_SETS = {
    "S0_physics_only": ["Y_phys_A", "Y_phys_B"],
    "S1_collapsed": ["log_tau", "inlet_temperature_K", "jacket_temperature_K",
                     "concentration_mol_L"],
    "S2_core_physics": ["log_tau", "inlet_temperature_K",
                        "jacket_temperature_K", "concentration_mol_L",
                        "T_eff", "log_Da1", "log_Da2", "log_k_ratio",
                        "Y_phys_A", "Y_phys_B"],
    "S3_physics_plus": ["log_tau", "inlet_temperature_K",
                        "jacket_temperature_K", "concentration_mol_L",
                        "T_eff", "b_number", "log_Da1", "log_Da2",
                        "log_k_ratio", "log_tau_over_tau_opt", "Y_max",
                        "conversion_X", "Y_phys_A", "Y_phys_B", "T_rise",
                        "delta_T"],
}


def get_set(X: pd.DataFrame, name: str) -> pd.DataFrame:
    return X[FEATURE_SETS[name]]
