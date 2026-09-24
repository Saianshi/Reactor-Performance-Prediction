"""Phase 2b -- competing physical explanations for what the base model misses.

The base Level-A model fits to RMSE 12.4 (CV 15.1). That is respectable but
not decisive, and it needs a NEGATIVE gamma (-3.33 K.L/mol, i.e. endothermic)
to use the concentration signal we proved is real. An endothermic A->B in a
jacket-HEATED reactor is possible, but it is also exactly what a misspecified
model looks like when it is handed a variable it cannot otherwise explain.

So gamma is only one of three candidate explanations for a genuine CA0
dependence, and they make different predictions. Test all three:

  H1  THERMAL       heat of reaction shifts T, and the shift scales with CA0.
                    -> the base model, gamma free.

  H2  KINETIC ORDER the reactions are not first order. With rate = k1*CA^n1,
                    writing a = CA/CA0 gives  da/dt = -k1*CA0^(n1-1)*a^n1, so
                    CA0 enters through CA0^(n1-1) and vanishes iff n1 = 1.
                    This is the hypothesis the original brief raised.

  H3  MIXING        the reactor is not ideal plug flow. The organisers said
                    "boundary value problem", and a PFR is an INITIAL value
                    problem -- axial dispersion is what makes it a BVP. Tested
                    here in its equivalent tanks-in-series form, which is
                    algebraic and needs no boundary iteration.

H2 and H3 are not free parameters bolted on to rescue a fit; each is a
specific, named departure from the textbook model with its own signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import physics as ph
from physics import _dexp, _phi, arrhenius, beta_of_Q

EXT_NAMES = ph.PARAM_NAMES + ["n1", "n2"]


@dataclass
class ExtParams(ph.PhysParams):
    """PhysParams plus reaction orders. n1 = n2 = 1 recovers the base model."""
    n1: float = 1.0
    n2: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in EXT_NAMES], float)

    @staticmethod
    def from_array(x) -> "ExtParams":
        return ExtParams(**dict(zip(EXT_NAMES, np.asarray(x, float))))


A_FLOOR = 1e-12


# ==========================================================================
# H2 -- arbitrary reaction orders, integrated stably
# ==========================================================================
def level_b_orders(p: ExtParams, df: pd.DataFrame, n_steps: int = 200,
                   full: bool = False):
    """Plug flow with rate1 = k1*CA^n1 and rate2 = k2*CB^n2.

    The exact-exponential trick still applies. Write the nonlinear rate as a
    first-order rate with a state-dependent coefficient,

        k1_eff = k1 * CA0^(n1-1) * a^(n1-1) ,

    freeze that coefficient over the step, and propagate the resulting LINEAR
    system exactly. The scheme therefore stays unconditionally stable at any
    Damkohler number and collapses to the exact first-order solver when
    n1 = n2 = 1, so H2 is nested inside the base model rather than being a
    different code path.
    """
    Q, CA0, T0, L, Tj = ph._unpack(df)
    tau = L / Q
    beta = beta_of_Q(p.log_beta0, p.n_flow, Q)
    gCA0 = p.gamma * CA0
    dt = tau / n_steps

    a = np.ones_like(tau)
    b = np.zeros_like(tau)
    T = T0.copy()
    h = 0.5 * beta * dt
    decay_half, phi_half = np.exp(-h), _phi(h)
    decay_full, phi_full = np.exp(-2 * h), _phi(2 * h)
    c1 = CA0 ** (p.n1 - 1.0)
    c2 = CA0 ** (p.n2 - 1.0)

    for _ in range(n_steps):
        T_old = T
        k1 = arrhenius(p.lnk1_ref, p.Ea1, T_old) * c1 * \
            np.maximum(a, A_FLOOR) ** (p.n1 - 1.0)
        T_mid = Tj + (T_old - Tj) * decay_half + \
            gCA0 * k1 * a * (0.5 * dt) * phi_half

        k1 = arrhenius(p.lnk1_ref, p.Ea1, T_mid) * c1 * \
            np.maximum(a, A_FLOOR) ** (p.n1 - 1.0)
        k2 = arrhenius(p.lnk2_ref, p.Ea2, T_mid) * c2 * \
            np.maximum(b, A_FLOOR) ** (p.n2 - 1.0)
        u, v = k1 * dt, k2 * dt
        a_new = a * np.exp(-u)
        b = b * np.exp(-v) + a * k1 * dt * _dexp(u, v)
        released = gCA0 * (a - a_new)
        a = np.clip(a_new, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        T = Tj + (T_old - Tj) * decay_full + released * phi_full

    Y = np.clip(100.0 * b, 0.0, 100.0)
    return {"Y": Y, "T_exit": T, "a_exit": a} if full else Y


# ==========================================================================
# H3 -- tanks in series (the algebraic equivalent of axial dispersion)
# ==========================================================================
def level_cstr_series(p: ph.PhysParams, df: pd.DataFrame, n_tanks: int = 10,
                      n_iter: int = 6, full: bool = False):
    """N equal CSTRs in series. N = 1 is a single stirred tank, N -> inf is a PFR.

    Each tank is a steady-state mass and energy balance, which is ALGEBRAIC,
    not differential:

        a_i = a_{i-1} / (1 + k1 t_i)
        b_i = (b_{i-1} + t_i k1 a_i) / (1 + k2 t_i)
        T_i = (T_{i-1} + t_i (beta Tj + gamma CA0 k1 a_i)) / (1 + beta t_i)

    with t_i = tau/N. The rate constants depend on T_i, which depends on the
    rates, so each tank is solved by a short fixed-point iteration. The
    dispersion number and N are related by Pe ~ 2(N-1), so scanning N is a
    scan over degree of backmixing.

    Every division here is by (1 + non-negative), so the recursion is
    unconditionally stable at any Damkohler number -- the same robustness
    property as the exponential integrator, for the same underlying reason.
    """
    Q, CA0, T0, L, Tj = ph._unpack(df)
    tau = L / Q
    beta = beta_of_Q(p.log_beta0, p.n_flow, Q)
    ti = tau / n_tanks

    a = np.ones_like(tau)
    b = np.zeros_like(tau)
    T = T0.copy()
    for _ in range(n_tanks):
        a_prev, b_prev, T_prev = a, b, T
        Ti = T_prev
        for _ in range(n_iter):
            k1 = arrhenius(p.lnk1_ref, p.Ea1, Ti)
            a_i = a_prev / (1.0 + k1 * ti)
            Ti = (T_prev + ti * (beta * Tj + p.gamma * CA0 * k1 * a_i)) / \
                 (1.0 + beta * ti)
            Ti = np.clip(Ti, 200.0, 1500.0)
        k1 = arrhenius(p.lnk1_ref, p.Ea1, Ti)
        k2 = arrhenius(p.lnk2_ref, p.Ea2, Ti)
        a = a_prev / (1.0 + k1 * ti)
        b = (b_prev + ti * k1 * a) / (1.0 + k2 * ti)
        T = Ti

    Y = np.clip(100.0 * b, 0.0, 100.0)
    return {"Y": Y, "T_exit": T, "a_exit": a} if full else Y


# ==========================================================================
# Fitting wrappers
# ==========================================================================
BOUNDS_EXT = ph.BOUNDS_FULL + [(0.3, 2.5), (0.3, 2.5)]      # n1, n2


def fit_ext(df, y, kind: str, fixed: dict | None = None, seed: int = 42,
            maxiter: int = 80, popsize: int = 14, n_tanks: int = 10,
            verbose: bool = False):
    """Global fit of an extended model. kind in {'orders', 'tanks'}."""
    from scipy.optimize import differential_evolution, least_squares

    fixed = fixed or {}
    if kind == "orders":
        bounds = list(BOUNDS_EXT)
        names = EXT_NAMES
        mk = ExtParams.from_array

        def predict(p, d):
            return level_b_orders(p, d, n_steps=200)
    elif kind == "tanks":
        bounds = list(ph.BOUNDS_FULL)
        names = ph.PARAM_NAMES
        mk = ph.PhysParams.from_array

        def predict(p, d):
            return level_cstr_series(p, d, n_tanks=n_tanks)
    else:
        raise ValueError(kind)

    def apply_fixed(p):
        for k, v in fixed.items():
            setattr(p, k, v)
        return p

    def obj(x):
        p = apply_fixed(mk(x))
        try:
            pred = predict(p, df)
        except (FloatingPointError, OverflowError, ValueError):
            return 1e6
        return 1e6 if not np.all(np.isfinite(pred)) else ph.rmse(pred, y)

    de = differential_evolution(obj, bounds, seed=seed, maxiter=maxiter,
                                popsize=popsize, tol=1e-10,
                                mutation=(0.4, 1.0), recombination=0.85,
                                polish=True, init="sobol",
                                updating="immediate", workers=1)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    ls = least_squares(lambda x: np.nan_to_num(
        predict(apply_fixed(mk(x)), df) - y, nan=1e3),
        np.clip(de.x, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
        max_nfev=200, xtol=1e-12, ftol=1e-12)
    p_de, p_ls = apply_fixed(mk(de.x)), apply_fixed(mk(ls.x))
    r_de, r_ls = ph.rmse(predict(p_de, df), y), ph.rmse(predict(p_ls, df), y)
    if verbose:
        print(f"        DE {r_de:.4f} -> LSQ {r_ls:.4f}")
    return (p_ls, r_ls) if r_ls <= r_de else (p_de, r_de)
