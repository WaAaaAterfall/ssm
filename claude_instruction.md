# Goal

I need you to implement a customized Poisson linear dynamical system (PLDS) on top of the `ssm` library (lindermanlab/ssm). You are running inside a checkout of that repo, so you have access to the source. I will paste the model specification, data interface, and design strategy below. The most important meta-point: **most of the "customization" is actually preprocessing, not new model code.** Please read the ssm source before writing anything, then implement.

# Scientific context (brief)

I have multielectrode recordings from a mouse forelimb reaching task. On each trial, neural spikes and 3D hand kinematics are recorded. On a subset of bins within each trial, a proprioceptive perturbation is applied — encoded by a known binary indicator `u_t ∈ {0, 1}`. I want to quantify how the perturbation changes the way kinematics drive the population's latent neural state. I have already fit a Poisson GLM with this structure; the PLDS is meant to be a more statistically efficient population-level version. Do not modify the science; just implement the model.

# Model specification

Let:
- `s_t ∈ R^p` be the latent neural state (p is a hyperparameter, try p ∈ {5, 10, 15, 20})
- `z_t ∈ R^{d_z}` be the (observed) kinematic vector at bin t, d_z = 6 (3D pos + 3D vel)
- `x_t ∈ Z_{≥0}^{d_x}` be the binned spike counts, d_x = 90 neurons
- `u_t ∈ {0, 1}` be the known perturbation indicator
- `Δ = 5/240` seconds, the bin width
- J be the kinematic lag order (try J ∈ {2, 3, 4})

State equation (linear-Gaussian, input-driven):

    s_{t+1} = F s_t
              + sum_{ℓ=0..J} G_ℓ z_{t-ℓ}
              + sum_{ℓ=0..J} u_{t-ℓ} * δG_ℓ z_{t-ℓ}
              + m_s
              + ε_t,
    ε_t ~ N(0, Q)

Observation equation (Poisson, log link):

    x^{(i)}_t ~ Poisson(Δ * exp(c_i^T s_t + b_i))

Parameters:
- F: p×p, latent state-transition
- Q: p×p, innovation covariance
- G_ℓ: p×d_z, baseline kinematic input (one per lag, ℓ = 0..J)
- δG_ℓ: p×d_z, **perturbation-modulated** kinematic input (one per lag) — primary object of scientific interest
- m_s: p-vector, latent offset
- C = [c_1, ..., c_{d_x}]^T: d_x×p, neural loading matrix
- b: d_x-vector, per-neuron log-rate baseline

# The key design insight (very important)

The gated input `u_{t-ℓ} * z_{t-ℓ}` is **a known, deterministic time series**, not a special model component. From the LDS's perspective it's just another input column. Therefore, **do not modify the LDS model class itself.** Instead, build an augmented input array as preprocessing:

    tilde_u_t = [ z_t, z_{t-1}, ..., z_{t-J},
                  u_t * z_t, u_{t-1} * z_{t-1}, ..., u_{t-J} * z_{t-J} ]
              ∈ R^{2*(J+1)*d_z}

and feed `tilde_u` to a standard input-driven Poisson LDS in `ssm`. The library's built-in input matrix B will then represent the concatenation [G_0, ..., G_J, δG_0, ..., δG_J]. You unpack B into G_ℓ and δG_ℓ after fitting.

This means: **no subclassing of the model is needed for the basic fit.** Only the per-block ridge penalty (see below) requires customization, and only if I want different shrinkage on G vs. δG.

# Files to read in the ssm repo FIRST, before writing anything

Read these and understand the API before coding:

1. `ssm/lds.py` — the LDS class. Confirm the constructor signature, the `emissions` and `dynamics` arguments, the `M` (input dimension) argument, the `fit` method's signature including how to pass inputs, and what method options exist for Poisson emissions.
2. `ssm/dynamics.py` — the `GaussianDynamics` (or whatever the input-driven Gaussian dynamics class is called) class. Look at its `m_step` method and how it handles inputs. Note the structure of parameters `As`, `bs`, `Vs`, etc. — figure out which one corresponds to F, which to the input matrix B, which to the offset m_s.
3. `ssm/emissions.py` — the Poisson emissions class. Confirm it does NOT need customization for our case.
4. `ssm/variational.py` (or similar) — to confirm what variational/Laplace inference option to use.
5. `notebooks/` — look for any Poisson LDS example or input-driven LDS example. There should be at least one demonstrating the input-driven fit on Poisson observations.

Report what you find before writing code. Specifically I need to know:
- The exact `LDS(...)` constructor signature for Poisson emissions with inputs
- The exact `fit(...)` signature and what `method=` options exist (we want Laplace-EM or its equivalent)
- The exact attribute names on the dynamics object that correspond to F, B, m_s, Q
- Whether `ssm` supports specifying a prior on the dynamics parameters, and if so, in what form

# Data interface

The user (i.e., me) supplies the data as three Python lists, one entry per trial:

- `z_list`: list of length N_trials; each entry is `np.ndarray` shape `(T_n, 6)` with `dtype=float`
- `x_list`: list of length N_trials; each entry is `np.ndarray` shape `(T_n, 90)` with non-negative integer counts
- `u_list`: list of length N_trials; each entry is `np.ndarray` shape `(T_n,)` with values in `{0, 1}` (a 1-D vector)

`T_n` may vary across trials. Treat each trial as independent (no information shared across trials in time).

# Tasks (in order)

## Task 1: Build the augmented input

Write a function:

```python
def build_augmented_inputs(z_list, u_list, J):
    """
    Construct the augmented input arrays (lagged kinematics + gated lagged kinematics)
    for an input-driven LDS. Pad/truncate trials appropriately so that the first J bins
    of each trial are dropped (consistent with my existing GLM convention).

    Returns:
        u_aug_list: list of length N_trials, each (T_n - J, 2*(J+1)*d_z) input arrays
        x_trim_list: list, x truncated to match u_aug_list time alignment
        z_trim_list: list, z truncated to match u_aug_list time alignment
        meta: dict with at least 'J', 'd_z', 'block_split' so we can later split B into G and δG.
    """
```

Note: there are several reasonable choices for handling lags. Pick the one that's most compatible with `ssm`'s expected per-trial array layout (i.e., all per-trial arrays must have the same time dimension within a trial). Document your choice in the docstring.

## Task 2: Implement a basic fit using off-the-shelf ssm

Write `fit_perturbation_plds(z_list, x_list, u_list, *, p, J, dt=5/240, n_iters=100, method=...)`:
- Construct augmented inputs via `build_augmented_inputs`
- Instantiate `ssm.LDS(N=d_x, D=p, M=2*(J+1)*d_z, emissions="poisson", dynamics="gaussian")` (verify exact API from your earlier reading)
- Initialize sensibly (the library's default may be poor for sparse Poisson data — if so, override with factor-analysis init on `log1p(x)` for C, b)
- Fit with Laplace-EM (or whatever ssm calls it)
- After fitting, extract and unpack:
  - F, m_s, Q from the dynamics parameters
  - B from the input matrix, split into G_0, ..., G_J, δG_0, ..., δG_J using meta['block_split']
  - C, b from the emissions
- Return a results dict with explicit keys: `'F'`, `'Q'`, `'m_s'`, `'G_lag'` (shape `(J+1, p, d_z)`), `'dG_lag'` (shape `(J+1, p, d_z)`), `'C'` (shape `(d_x, p)`), `'b'` (shape `(d_x,)`), plus `'p'`, `'J'`, `'dt'`, `'log_marginal_likelihood_trace'`, and a reference to the fitted ssm object as `'_ssm_model'`.

Also include the per-block ridge as a hyperparameter (`lam_G`, `lam_dG`). For Task 2, use the same penalty on both (`lam_G = lam_dG`) — this works with ssm's default priors if any. Task 4 below adds the per-block customization.

## Task 3: Implement evaluation

Write `evaluate_perturbation_plds(results, z_list, x_list, u_list)`:
- Run the smoother to get smoothed latents `E[s_t]` per trial
- Compute one-step-ahead predicted firing rates `μ_t = Δ * exp(C @ s_pred_t + b)` where `s_pred_t = F @ E[s_{t-1}] + (input drive at t)`
- Return a metrics dict matching my existing GLM evaluator: `deviance_x`, `deviance_x_per_obs`, `loglik_x`, `pseudoR2_x` (using a constant-rate null), and additionally `held_out_marginal_likelihood`.

The Poisson deviance formulas should match those in my GLM code:

```python
def _poisson_deviance(y_true, mu, eps=1e-9):
    mu = np.clip(mu, eps, None)
    yl = np.where(y_true > 0, y_true * np.log((y_true + eps) / mu), 0.0)
    return 2.0 * np.sum(yl - (y_true - mu))
```

## Task 4: The one real customization — per-block ridge

Read the dynamics class in `ssm/dynamics.py` and the m_step. Implement a subclass `PerBlockRidgeGaussianDynamics` (or whatever fits the ssm naming convention) that overrides `m_step` to apply different ridge penalties to different column blocks of the input matrix B. Specifically:
- Columns 0..((J+1)*d_z - 1) are the G block, penalized at strength `lam_G`
- Columns ((J+1)*d_z)..(2*(J+1)*d_z - 1) are the δG block, penalized at strength `lam_dG`
- F is penalized at strength `lam_F` (can be 0 by default — F is well-constrained from data)
- m_s is unpenalized

The M-step normal equations are standard: given smoothed expectations `E[s_{t+1} r_t^T]` and `E[r_t r_t^T]` where `r_t = [s_t; tilde_u_t; 1]`,

    Theta_hat = (sum_t E[s_{t+1} r_t^T]) @ inv(sum_t E[r_t r_t^T] + Lambda)

where `Lambda` is a diagonal of per-block penalties (zeros in the F block if `lam_F=0`, zero in the intercept column, `lam_G` for G columns, `lam_dG` for δG columns). Q is then estimated from residuals as usual. The expectations come straight from ssm's E-step (use whatever internal API the existing GaussianDynamics m_step uses to access them — read it carefully).

Wire this subclass in by modifying `fit_perturbation_plds` to optionally pass `dynamics=PerBlockRidgeGaussianDynamics(...)` when `lam_G != lam_dG` (or unconditionally — your choice).

## Task 5: Nested models for hypothesis testing

Implement two ablations as separate fit functions:

- `fit_perturbation_plds_no_dG(...)`: identical, but δG_ℓ = 0 fixed. Implement by constructing the augmented input WITHOUT the gated columns (just `[z_t, z_{t-1}, ..., z_{t-J}]`).
- `fit_perturbation_plds_intrinsic(...)`: no kinematic inputs at all (autonomous PLDS). Easiest: construct augmented input as `np.zeros((T_n - J, 1))` or use `M=0` if ssm supports it.

Both should return results dicts in the same format as Task 2 (with absent keys set to `None` or zero arrays of the right shape).

## Task 6: Cross-validation

Write `crossval_perturbation_plds(z_list, x_list, u_list, *, p, J, n_folds=5, lam_G, lam_dG, ...)`:
- Split trials (NOT bins) into n_folds. Important: split by *trial index*, never by within-trial bin index.
- For each fold, fit on train trials, evaluate on held-out trials
- Return per-fold metrics + mean/std

Run this for both the full model and the two nested ablations, so the user can compare held-out `deviance_x` across the three models — this is the actual scientific test of whether δG is real.

## Task 7: Synthetic-data smoke test

Generate synthetic data from the model with **known** F, G_ℓ, δG_ℓ, C, b. Use:
- p_true = 4, p_fit = 6 (test mild overparameterization)
- J = 2
- N_trials = 50, T = 160
- d_x = 60 (smaller for fast iteration), d_z = 6
- Use realistic firing rates: per-neuron mean count per bin in [0.05, 0.5] (otherwise the recurrent signal is unidentifiable; see notes in my codebase)
- Sample u_t as Bernoulli(0.3) per bin
- True δG_ℓ should be non-zero and structured (e.g., scaled random matrix)

Verify:
1. Fitting completes without error and the marginal likelihood is monotone non-decreasing across EM iterations.
2. The reconstructed F has similar dominant eigenvalues to the true F (up to gauge — recall the latent state is identified only up to invertible linear transform; eigenvalues of F are invariant).
3. The per-trial smoothed latents, when affinely regressed onto the true latents, give R² > ~0.7.
4. Critically: **fitting on data with δG_true = 0 should give held-out deviance no better for the full model than for the no-dG model.** Conversely, with δG_true ≠ 0, the full model should win on held-out data. This is the model-selection sanity check.

# Things NOT to do

1. **Do not re-implement Kalman filtering, smoothing, or Laplace-EM from scratch.** Use ssm's existing inference. The whole point of using this library is to inherit its battle-tested inference.
2. **Do not modify the Poisson emissions class.** Off-the-shelf Poisson emissions are exactly what we need.
3. **Do not treat u_t as a latent or stochastic variable.** It is known data, deterministically observed.
4. **Do not split CV folds by bin.** Always split by trial.
5. **Do not write your own EM loop.** Call ssm's `fit` method.
6. **Do not change anything in the ssm library itself**, only add new files in a new subdirectory I'll specify or in `examples/` if conventional. The subclassed dynamics class lives in my new module, not in ssm core.

# Deliverables

Place all new code in a new directory `perturbation_plds/` at the repo root:

```
perturbation_plds/
├── __init__.py
├── inputs.py         # build_augmented_inputs and related preprocessing
├── model.py          # PerBlockRidgeGaussianDynamics subclass
├── fit.py            # fit_perturbation_plds and the two ablation fitters
├── evaluate.py       # evaluate_perturbation_plds with Poisson deviance
├── crossval.py       # crossval_perturbation_plds
├── synthetic.py      # generate_synthetic_data and the smoke test
└── README.md         # explain the model, the preprocessing trick, and how to use
```

Each function should have a docstring with shapes. Where you make a non-obvious choice (e.g., how to handle the first J bins of each trial; how to initialize when ssm's defaults are poor) document it briefly in the docstring AND in the README.

Run the synthetic smoke test before declaring done; report the output. If anything fails or the sanity checks don't pass, stop and tell me what you found rather than papering over it.

# Process

1. First, read the ssm source files I listed and summarize what you found (API signatures, parameter names, init options). Confirm the basic plan still works given the actual API.
2. Then implement Task 1 and Task 2 (basic fit). Run a tiny sanity check on toy data.
3. Then Task 3 (eval), Task 5 (ablations), Task 6 (CV). These are mostly orchestration.
4. Then Task 4 (per-block ridge subclass). This is the trickiest — read the existing m_step carefully.
5. Finally Task 7 (full synthetic smoke test).

If you hit a place where ssm's API doesn't support what we need cleanly, **stop and tell me** rather than working around it with a wrapper hack. There might be a way I haven't thought of, or it might genuinely require dropping to dynamax instead.