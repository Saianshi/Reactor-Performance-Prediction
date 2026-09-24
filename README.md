# Reactor Performance Prediction

An ML hackathon entry that predicts the yield of a non-isothermal plug-flow reactor running a series reaction A -> B -> C, given only 150 training rows. Instead of throwing a generic regressor at five raw columns, the model is built around an actual physics prior (Arrhenius kinetics fit to the data) and lets a Gaussian process learn only the small correction on top of it. With this few rows, that turned out to matter a lot.

## The problem

Given `flow_rate_L_min`, `concentration_mol_L`, `inlet_temperature_K`, `length_m`, and `jacket_temperature_K` for a continuous flow reactor, predict `overall_yield`, the final yield of product B. The full brief is in `6a633c384e7eb_ML_Hackathon_Problem_Statement_Final.pdf`.

## Approach

1. **Fit the physics first.** `src/physics.py` integrates the non-isothermal plug-flow ODEs for the series reaction, with Arrhenius rate constants for both steps. The reactor's cross-sectional area is unknown, so a residence-time proxy `t = L/Q` is used instead of the true residence time `tau = V/Q`; the unknown area constant gets absorbed into the fitted pre-exponential factors, so nothing is lost.
2. **Test whether L and Q really need to be separate inputs.** `src/collapse_test.py` and `src/collapse_decisive.py` run three independent tests (LOO across coordinate systems, a residual-information test, and a matched-pair test) to check whether L and Q act only through the ratio `tau = L/Q`, which would mean the true response surface is 3-D, not 5-D. The result decided which features actually get built in `src/features.py` rather than just asserting an answer.
3. **Feature engineering, not polynomial expansion.** Every feature in `src/features.py` is either a raw input, a dimensionless group with an actual physical meaning, or an output of the fitted reactor model. With only 150 rows, a blind polynomial basis would overfit before it explained anything.
4. **Model zoo, fit inside every fold.** `src/models.py` defines a set of candidates, headlined by `gp_residual`: a Gaussian process whose prior mean is the physics prediction, so it only has to learn a smooth correction rather than the whole response surface from scratch. `src/run_models.py` refits the physics itself inside every cross-validation fold (not just the regressor), which is the expensive but correct way to do it, since the physics parameters are fit against the training target too.
5. **Validation discipline.** `src/validate.py` enforces one rule: nothing that touches the target, not the scaler, not the feature selection, not the physics, gets fit outside the fold. Repeated 10x10 K-fold and leave-one-out are both reported, with the repeat-to-repeat spread labelled separately from sampling error so the two aren't confused.
6. **Identifiability check.** `src/profile_likelihood.py` profiles the RMSE surface around each fitted physics parameter to check they're actually pinned down by the data rather than sitting at an optimiser bound. One activation energy (Ea2) originally pinned itself at the search bound in every fold; the bound was widened and refit rather than reported as-is.
7. **Finalize.** `src/finalize.py` refits the chosen model once on all 150 rows, runs a robustness battery on the test predictions, and writes the submission file. It reopens no modelling decisions, only applies whatever cross-validation already chose.

## Results

- Predict-the-mean baseline (target standard deviation): RMSE 38.3.
- Physics-only model, 10-fold CV: RMSE 6.88 (down from an early bound-pinned fit of 12.2 once the activation-energy search range was widened).
- Fitted physics parameters are well identified: profiling each one around its optimum shows a narrow, well-defined minimum rather than a flat ridge (see `outputs/figures/04_profile_likelihood.png`).
- Full physics-plus-GP cross-validation numbers, fold-by-fold, are in `outputs/cv/`; the raw run logs (`phase2_log.txt`, `phase2c_log.txt`, `profile_log.txt`) show the actual optimizer trace, not just the summary.

## Project layout

```
data/                       train/test CSVs used by the pipeline (duplicated at repo root
                            as originally provided)
src/config.py               paths, seed, central config everything else imports from
src/physics.py              the plug-flow ODE integrator and Arrhenius kinetics
src/fit_physics.py          fits and cross-validates the physics model variants
src/collapse_test.py        tests whether L, Q collapse to tau = L/Q
src/collapse_decisive.py    paired, error-barred follow-up on the same question
src/test_hypotheses.py      adjudicates thermal vs kinetic vs mixing explanations
                            for the observed concentration dependence
src/features.py             physically-motivated feature engineering
src/models.py               the model zoo, headlined by a physics-prior GP
src/run_models.py           evaluates every model under one fold-safe protocol
src/validate.py             repeated k-fold, leave-one-out, nested CV
src/profile_likelihood.py   checks the fitted physics parameters are identifiable
src/finalize.py             refits the chosen model once and writes the submission
outputs/                    figures, per-fold CV results, run logs, fitted parameters
KGPian_Algorithms.csv       the submitted predictions (overall_yield for the 50 test rows)
```

## Running it

```bash
pip install numpy pandas scipy scikit-learn matplotlib

cd src
python fit_physics.py          # fits and cross-validates the physics model
python collapse_test.py        # dimension-collapse test (L, Q vs tau)
python run_models.py           # cross-validates the full model zoo
python profile_likelihood.py   # checks parameter identifiability
python finalize.py --team KGPian_Algorithms
```

Each script writes its results to `outputs/`, so later scripts can be re-run independently once the earlier ones have produced their output files.
