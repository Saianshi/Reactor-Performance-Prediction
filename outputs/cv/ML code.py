import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except ImportError:
    print("[!] pip install tqdm  -- fallback chal raha hai")
    def tqdm(x=None, **kw): return x if x is not None else []

BASE       = r"D:\1SM task\IIT khagpur"
TRAIN      = os.path.join(BASE, "train_dataset.csv")
TEST       = os.path.join(BASE, "test_dataset.csv")
TEAM_NAME  = "KGPian_Algorithms"       # <-- apna team name
OUT        = os.path.join(BASE, f"{TEAM_NAME}.csv")

N_FOLDS        = 20        # 20-fold
N_REPEATS      = 3         # x3 repeats  => 60 train/valid cycles
GLOBAL_RESTART = 250       # physics param global search restarts (SLOW, ~5-10 min)
FOLD_RESTART   = 6         # per-fold physics refinement restarts
RK4_STEPS      = 160       # ODE integration steps along reactor length
SEED           = 42

FEATS = ['flow_rate_L_min','concentration_mol_L','inlet_temperature_K',
         'length_m','jacket_temperature_K']

C = dict(g="\033[92m", y="\033[93m", c="\033[96m", b="\033[1m",
         d="\033[90m", r="\033[91m", x="\033[0m")
if os.name == "nt": os.system("")
def rule(ch="─", n=82): print(C['d'] + ch*n + C['x'])
def header(t):
    rule("═"); print(f"{C['b']}{C['c']}  {t}{C['x']}"); rule("═")
def bar(f, w=30):
    f = max(0.0, min(1.0, float(f)));  k = int(round(f*w))
    return "█"*k + "░"*(w-k)

T_START = time.time()

def pfr_simulate(theta, Q, CA0, T0, L, Tj, n_steps=RK4_STEPS):
    """Vectorised fixed-step RK4 over all samples simultaneously."""
    lnA1, E1, lnA2, E2, J1, J2, U, lnAc = theta
    tau = np.exp(lnAc) * L / Q
    h   = tau / n_steps
    CA, CB, T = CA0.copy(), np.zeros_like(CA0), T0.copy()

    def f(CA, CB, T):
        Tc = np.clip(T, 150.0, 2000.0)
        k1 = np.exp(np.clip(lnA1 - E1/Tc, -40, 40))
        k2 = np.exp(np.clip(lnA2 - E2/Tc, -40, 40))
        r1, r2 = k1*CA, k2*CB
        return -r1, r1 - r2, J1*r1 + J2*r2 + U*(Tj - Tc)

    for _ in range(n_steps):
        a1,b1,c1 = f(CA, CB, T)
        a2,b2,c2 = f(CA+h/2*a1, CB+h/2*b1, T+h/2*c1)
        a3,b3,c3 = f(CA+h/2*a2, CB+h/2*b2, T+h/2*c2)
        a4,b4,c4 = f(CA+h*a3,   CB+h*b3,   T+h*c3)
        CA = np.maximum(CA + h/6*(a1+2*a2+2*a3+a4), 0.0)
        CB = np.maximum(CB + h/6*(b1+2*b2+2*b3+b4), 0.0)
        T  = np.clip(   T  + h/6*(c1+2*c2+2*c3+c4), 150.0, 2000.0)
    return 100.0 * CB / CA0          # yield % of B

LO = np.array([ -5.0,  1000.0,  -5.0,  1000.0, -200.0, -200.0, 0.0, -4.0])
HI = np.array([ 40.0, 20000.0,  40.0, 25000.0,  200.0,  200.0, 5.0,  4.0])
XSC = np.array([1, 1000, 1, 1000, 10, 10, 1, 1], dtype=float)

def fit_physics(Xraw, yv, n_restarts, rng, theta_seed=None, tag="physics", quiet=False):
    Q, CA0, T0, L, Tj = Xraw.T
    def resid(th):
        r = pfr_simulate(th, Q, CA0, T0, L, Tj) - yv
        return np.nan_to_num(r, nan=1e4, posinf=1e4, neginf=-1e4)

    starts = []
    if theta_seed is not None:
        starts.append(np.clip(theta_seed, LO, HI))
        for _ in range(max(0, n_restarts - 1)):                 # jitter around seed
            j = theta_seed * (1 + 0.15*rng.standard_normal(8)) + 0.05*rng.standard_normal(8)
            starts.append(np.clip(j, LO, HI))
    else:
        for _ in range(n_restarts):
            p = LO + rng.random(8)*(HI - LO)
            p[1] = rng.uniform(3000, 15000); p[3] = rng.uniform(3000, 20000)
            starts.append(p)

    best = (np.inf, None)
    it = tqdm(starts, desc=f"  {tag}", ncols=82,
              bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} {postfix}") if not quiet else starts
    for p0 in it:
        try:
            res = least_squares(resid, p0, bounds=(LO, HI), max_nfev=300, x_scale=XSC)
        except Exception:
            continue
        e = float(np.sqrt(np.mean(res.fun**2)))
        if e < best[0]:
            best = (e, res.x)
            if not quiet:
                try: it.set_postfix_str(f"best RMSE {e:.3f}")
                except Exception: pass
    return best[1], best[0]

# ==================================================================
#  PART B - FEATURE ENGINEERING (chemical-engineering driven)
# ==================================================================
def make_features(df, phys=None):
    d = pd.DataFrame(index=df.index)
    Q, CA = df['flow_rate_L_min'], df['concentration_mol_L']
    T0, L, Tj = df['inlet_temperature_K'], df['length_m'], df['jacket_temperature_K']
    d['Q'], d['CA0'], d['T0'], d['L'], d['Tj'] = Q, CA, T0, L, Tj

    tau = L / Q                                                  # residence time
    d['tau'], d['log_tau'], d['sqrt_tau'] = tau, np.log(tau), np.sqrt(tau)

    Tavg = 0.5*(T0 + Tj)                                         # thermal setpoint
    d['Tavg'], d['dT'] = Tavg, Tj - T0
    d['invT'], d['invT0'], d['invTj'] = 1000/Tavg, 1000/T0, 1000/Tj

    for E in [4000, 6000, 8000, 10000, 12000, 15000]:            # Damkohler family
        da = tau * np.exp(-E/Tavg)
        d[f'Da_{E}']    = da
        d[f'logDa_{E}'] = np.log(da + 1e-30)
        d[f'conv_{E}']  = 1.0 - np.exp(-np.clip(da*1e5, 0, 50))

    for dE in [2000, 4000, 6000]:                                # selectivity k2/k1
        d[f'sel_{dE}']     = np.exp(-dE/Tavg)
        d[f'sel_tau_{dE}'] = tau * np.exp(-dE/Tavg)

    d['heat_load']    = CA * tau
    d['heat_flux']    = (Tj - T0) * tau
    d['CA_over_Q']    = CA / Q
    d['LxCA']         = L * CA
    d['T_jacket_tau'] = Tj * tau

    if phys is not None:                                         # ODE surrogate output
        d['phys_yield'] = phys
        d['phys_sq']    = phys**2 / 100.0
    return d

# ==================================================================
#  1. LOAD
# ==================================================================
header("STAGE 1/6  ·  DATA LOADING")
train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
Xraw_tr, Xraw_te = train[FEATS].values, test[FEATS].values
y = train['overall_yield'].values
print(f"  train  -> {C['g']}{train.shape[0]} x {train.shape[1]}{C['x']}   "
      f"test -> {C['g']}{test.shape[0]} x {test.shape[1]}{C['x']}   "
      f"missing: {train.isna().sum().sum()}/{test.isna().sum().sum()}")
print(f"  target : min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}  std={y.std():.2f}")

# ==================================================================
#  2. GLOBAL PHYSICS PARAMETER IDENTIFICATION  (the slow, useful part)
# ==================================================================
header(f"STAGE 2/6  ·  PHYSICS ODE PARAMETER IDENTIFICATION  ({GLOBAL_RESTART} restarts)")
print(f"  {C['d']}Fitting 8 kinetic/thermal params by multi-start Levenberg-Marquardt.{C['x']}")
print(f"  {C['d']}Har restart = 300 ODE-solve iterations x 150 samples. Time lagega.{C['x']}\n")
rng = np.random.default_rng(SEED)
theta_glob, rmse_glob = fit_physics(Xraw_tr, y, GLOBAL_RESTART, rng, tag="global search")
names = ['lnA1','E1/R','lnA2','E2/R','J1','J2','U','ln(Ac)']
print(f"\n  {C['b']}Identified parameters:{C['x']}")
for n_, v_ in zip(names, theta_glob):
    print(f"    {n_:<8s} = {v_:12.4f}")
print(f"  {C['g']}in-sample physics RMSE = {rmse_glob:.4f}{C['x']}   "
      f"(elapsed {time.time()-T_START:.0f}s)")

# ==================================================================
#  3. FEATURES
# ==================================================================
header("STAGE 3/6  ·  FEATURE ENGINEERING")
phys_tr_full = pfr_simulate(theta_glob, *Xraw_tr.T)
phys_te_full = pfr_simulate(theta_glob, *Xraw_te.T)
Xtr_df = make_features(train[FEATS], phys_tr_full)
n_feat = Xtr_df.shape[1]
print(f"  raw={len(FEATS)}  ->  engineered={C['g']}{n_feat}{C['x']}")
corr = pd.concat([Xtr_df, pd.Series(y, name='y', index=Xtr_df.index)], axis=1).corr()['y'].drop('y')
print(f"\n  {C['b']}Top-10 by |corr|:{C['x']}")
for f_, v_ in corr.abs().sort_values(ascending=False).head(10).items():
    print(f"    {f_:<16s} {bar(abs(v_))} {corr[f_]:+.3f}")

# ==================================================================
#  4. HYPERPARAMETER SEARCH (inner 5-fold)
# ==================================================================
header("STAGE 4/6  ·  HYPERPARAMETER SEARCH (inner 5-fold)")
Xfull = Xtr_df.values
inner = KFold(5, shuffle=True, random_state=SEED)

def inner_cv(model_fn, X, yv):
    errs = []
    for tri, vai in inner.split(X):
        m = model_fn(); m.fit(X[tri], yv[tri])
        errs.append(np.sqrt(mean_squared_error(yv[vai], np.clip(m.predict(X[vai]), 0, 100))))
    return float(np.mean(errs))

grids = {
 'ExtraTrees': [dict(n_estimators=e, min_samples_leaf=l, max_features=f)
                for e in (600, 1000) for l in (1, 2) for f in (0.5, 0.8, 1.0)],
 'HistGBM'   : [dict(max_iter=i, learning_rate=lr, max_leaf_nodes=n, l2_regularization=l2)
                for i in (500, 900) for lr in (0.03, 0.06) for n in (8, 15) for l2 in (0.5, 3.0)],
 'KRR'       : [dict(alpha=a, gamma=g) for a in (0.01, 0.1, 1.0) for g in (0.01, 0.05, 0.2, 0.5)],
 'SVR'       : [dict(C=c, gamma=g, epsilon=0.1) for c in (100, 1000, 5000) for g in (0.01, 0.05, 0.2)],
}
builders = {
 'ExtraTrees': lambda p: ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **p),
 'HistGBM'   : lambda p: HistGradientBoostingRegressor(random_state=SEED, **p),
 'KRR'       : lambda p: make_pipeline(StandardScaler(), KernelRidge(kernel='rbf', **p)),
 'SVR'       : lambda p: make_pipeline(StandardScaler(), SVR(kernel='rbf', **p)),
}
best_params = {}
for mname, grid in grids.items():
    best = (np.inf, None)
    pb = tqdm(grid, desc=f"  {mname:<11s}", ncols=82,
              bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} {postfix}")
    for p in pb:
        e = inner_cv(lambda: builders[mname](p), Xfull, y)
        if e < best[0]:
            best = (e, p)
            try: pb.set_postfix_str(f"best {e:.3f}")
            except Exception: pass
    best_params[mname] = best[1]
    print(f"    -> {mname:<11s} RMSE {C['y']}{best[0]:.3f}{C['x']}  {best[1]}")

def build_models():
    return {
      'GPR-Matern': make_pipeline(StandardScaler(),
          GaussianProcessRegressor(
              kernel=ConstantKernel(100.0)*Matern(length_scale=np.ones(n_feat), nu=2.5)
                     + WhiteKernel(noise_level=1.0),
              normalize_y=True, n_restarts_optimizer=4, random_state=SEED)),
      'ExtraTrees': builders['ExtraTrees'](best_params['ExtraTrees']),
      'HistGBM'   : builders['HistGBM'](best_params['HistGBM']),
      'RandForest': RandomForestRegressor(n_estimators=800, random_state=SEED, n_jobs=-1),
      'KRR'       : builders['KRR'](best_params['KRR']),
      'SVR'       : builders['SVR'](best_params['SVR']),
    }
MODELS = list(build_models().keys()) + ['PhysicsODE']

# ==================================================================
#  5. 20-FOLD x 3-REPEAT CROSS VALIDATION
#     Physics params are RE-FITTED inside every fold (no leakage).
# ==================================================================
header(f"STAGE 5/6  ·  {N_FOLDS}-FOLD x {N_REPEATS}-REPEAT CROSS VALIDATION "
       f"({N_FOLDS*N_REPEATS} cycles)")
rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
splits = list(rkf.split(Xraw_tr))
oof_sum = {m: np.zeros(len(y)) for m in MODELS}
oof_cnt = np.zeros(len(y))
fold_rmse = {m: [] for m in MODELS}
rng_f = np.random.default_rng(SEED + 1)
t_cv = time.time()

for i, (tri, vai) in enumerate(splits, 1):
    rep, fold = (i-1)//N_FOLDS + 1, (i-1) % N_FOLDS + 1
    print(f"\n{C['b']}┌─ REPEAT {rep}/{N_REPEATS}  FOLD {fold}/{N_FOLDS}{C['x']}"
          f"   train={len(tri)}  valid={len(vai)}")

    # -- physics refit on this fold's training data only --
    th_f, _ = fit_physics(Xraw_tr[tri], y[tri], FOLD_RESTART, rng_f,
                          theta_seed=theta_glob, quiet=True)
    ph_tr = pfr_simulate(th_f, *Xraw_tr[tri].T)
    ph_va = pfr_simulate(th_f, *Xraw_tr[vai].T)
    Xf_tr = make_features(train[FEATS].iloc[tri], ph_tr).values
    Xf_va = make_features(train[FEATS].iloc[vai], ph_va).values

    p_phys = np.clip(ph_va, 0, 100)
    oof_sum['PhysicsODE'][vai] += p_phys
    r_ph = np.sqrt(mean_squared_error(y[vai], p_phys))
    fold_rmse['PhysicsODE'].append(r_ph)
    print(f"│   {'PhysicsODE':<12s}  RMSE {C['c']}{r_ph:7.3f}{C['x']}   {C['d']}(ODE surrogate){C['x']}")

    fm = build_models()
    pbar = tqdm(list(fm.keys()), desc=f"  r{rep}f{fold}", ncols=82,
                bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} {postfix}", leave=False)
    for name in pbar:
        try: pbar.set_postfix_str(name)
        except Exception: pass
        t_m = time.time()
        m = fm[name]; m.fit(Xf_tr, y[tri])
        p = np.clip(m.predict(Xf_va), 0, 100)
        oof_sum[name][vai] += p
        r_ = np.sqrt(mean_squared_error(y[vai], p))
        fold_rmse[name].append(r_)
        print(f"│   {name:<12s}  RMSE {C['y']}{r_:7.3f}{C['x']}   "
              f"MAE {mean_absolute_error(y[vai], p):6.3f}   "
              f"R² {r2_score(y[vai], p) if len(vai) > 1 else float('nan'):+.4f}   "
              f"{C['d']}{time.time()-t_m:5.1f}s{C['x']}")
    oof_cnt[vai] += 1
    done = i/len(splits)
    eta = (time.time()-t_cv)/done*(1-done)
    print(f"{C['b']}└─{C['x']} |{C['g']}{bar(done, 44)}{C['x']}| {done*100:5.1f}%   "
          f"elapsed {time.time()-t_cv:6.1f}s   ETA {eta:6.1f}s")

oof = {m: oof_sum[m]/np.maximum(oof_cnt, 1) for m in MODELS}

# ==================================================================
#  6. STACKING + FINAL PREDICTION
# ==================================================================
print()
header("STAGE 6/6  ·  OOF SUMMARY, NNLS STACKING & SUBMISSION")
print(f"  {'MODEL':<13}{'OOF RMSE':>10}{'FOLD MEAN':>11}{'FOLD STD':>10}{'MAE':>9}{'R²':>9}")
rule()
for m in MODELS:
    o = oof[m]
    print(f"  {m:<13}{np.sqrt(mean_squared_error(y,o)):>10.3f}"
          f"{np.mean(fold_rmse[m]):>11.3f}{np.std(fold_rmse[m]):>10.3f}"
          f"{mean_absolute_error(y,o):>9.3f}{r2_score(y,o):>9.4f}")
rule()

A = np.column_stack([oof[m] for m in MODELS])
w, _ = nnls(A, y)                       # non-negative least squares stack
if w.sum() <= 0: w = np.ones(len(MODELS))
w = w / w.sum()
blend_oof = np.clip(A @ w, 0, 100)
print(f"\n  {C['b']}NNLS stacking weights:{C['x']}")
for m, wi in zip(MODELS, w):
    print(f"    {m:<13} {bar(wi)} {wi:.3f}")
best_single = min(np.sqrt(mean_squared_error(y, oof[m])) for m in MODELS)
print(f"\n  {C['b']}{C['g']}STACKED OOF RMSE = {np.sqrt(mean_squared_error(y, blend_oof)):.4f}{C['x']}"
      f"   (best single = {best_single:.4f})")

# ---- full-data refit ----
print()
final_models = build_models()
preds = {'PhysicsODE': np.clip(phys_te_full, 0, 100)}
Xte_full = make_features(test[FEATS], phys_te_full).values
for name in tqdm(list(final_models.keys()), desc="  final refit ", ncols=82,
                 bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt}"):
    final_models[name].fit(Xfull, y)
    preds[name] = np.clip(final_models[name].predict(Xte_full), 0, 100)

final = np.clip(sum(wi*preds[m] for wi, m in zip(w, MODELS)), 0.0, 100.0)

sub = pd.DataFrame({'overall_yield': np.round(final, 4)})
assert len(sub) == 50, f"[X] 50 rows chahiye, mile {len(sub)}"
assert list(sub.columns) == ['overall_yield'], "[X] header galat"
assert sub['overall_yield'].notna().all(), "[X] NaN in predictions"
sub.to_csv(OUT, index=False)

rule("═")
print(f"  {C['g']}✓ SUBMISSION SAVED{C['x']}  ->  {OUT}")
print(f"  rows={len(sub)}  min={final.min():.3f}  max={final.max():.3f}  "
      f"mean={final.mean():.3f}  std={final.std():.3f}")
print(f"  TOTAL RUNTIME: {time.time()-T_START:.1f}s")
rule("═")
print(sub.head(10).to_string(index=False))