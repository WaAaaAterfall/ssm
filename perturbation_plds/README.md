# Perturbation PLDS

A customized **Poisson Linear Dynamical System** built on top of
[`ssm`](https://github.com/lindermanlab/ssm) for quantifying how a
proprioceptive perturbation changes the way kinematics drive a population's
latent neural state.

## Model

Latent neural state `s_t ∈ R^p`, observed kinematics `z_t ∈ R^{d_z}` (d_z = 6),
spike counts `x_t ∈ Z_{≥0}^{d_x}` (d_x = 90), known binary perturbation `u_t`.

**State equation** (linear-Gaussian, input-driven):

```
s_{t+1} = F s_t + Σ_{ℓ=0..J} G_ℓ z_{t-ℓ}
              + Σ_{ℓ=0..J} u_{t-ℓ} · δG_ℓ z_{t-ℓ}
              + m_s + ε_t,        ε_t ~ N(0, Q)
```

**Observation equation** (Poisson, log link):

```
x^{(i)}_t ~ Poisson(Δ · exp(c_i^T s_t + b_i))
```

`δG_ℓ` (the perturbation-modulated kinematic gain) is the primary scientific
object of interest.

## The preprocessing trick (why almost no model code is needed)

The gated input `u_{t-ℓ} · z_{t-ℓ}` is a **known, deterministic** time series.
From the LDS's perspective it is just another input column. So we build an
augmented input

```
tilde_u_t = [ z_t, …, z_{t-J},  u_t·z_t, …, u_{t-J}·z_{t-J} ] ∈ R^{2(J+1)d_z}
```

and feed it to a standard input-driven Poisson LDS. ssm's input matrix
`B = dynamics.Vs[0]` then equals `[G_0 … G_J | δG_0 … δG_J]`, which we unpack
after fitting. **No subclassing is needed for the basic fit.**

### Parameter mapping to ssm

For an `ssm.LDS`, the dynamics are `AutoRegressiveObservations` with `K=1,
lags=1`:

| Model | ssm attribute |
|-------|---------------|
| `F`   | `dynamics.As[0]`     (p×p) |
| `B = [G | δG]` | `dynamics.Vs[0]` (p×M) |
| `m_s` | `dynamics.bs[0]`     (p,) |
| `Q`   | `dynamics.Sigmas[0]` (p×p) |
| `C`   | `emissions.Cs[0]`    (d_x×p) |
| `b`   | `emissions.ds[0]`    (d_x,) |

## Two deliberate customizations

Both live in `model.py`; **neither modifies the ssm library**.

1. **`NoInputPoissonEmissions`** — ssm's `PoissonEmissions` carry a *fitted*
   input matrix `Fs` (`forward = C·s + F·u + d`), and ssm feeds the same input
   array to both dynamics and emissions. The spec has observations depend
   **only** on the latent state, so we freeze `Fs = 0` (excluded from `params`,
   re-zeroed after init). This removes a direct kinematics→spikes path that
   would otherwise bypass — and compete with — the latent state. ssm's `LDS`
   shares one input dimension `M` across dynamics and emissions, so there is no
   built-in way to give emissions `M=0` while dynamics has `M>0`; freezing is
   the clean fix. *(Confirmed with the user before implementing.)*

2. **`PerBlockRidgeGaussianDynamics`** — per-block ridge on the input matrix.
   ssm's AR M-step already solves the regularized normal equations
   `W = (Σ E[r r^T] + J0)^{-1}(Σ E[s' r^T] + h0)^T` with `r=[s; tilde_u; 1]`. We
   set the diagonal prior precision `J0` to `lam_F` on the F columns, `lam_G` on
   the G columns, `lam_dG` on the δG columns, `0` on the bias, with `h0=0`
   (pure ridge shrinkage toward zero).

   **Important:** for a plain `AutoRegressiveObservations` with `lags==1`, ssm
   runs an *exact* M-step that **ignores** `J0`/`h0`. ssm selects that path via
   an exact `type(...) in [...]` check, so our subclass is *not* matched and
   instead uses the prior-respecting M-step. We therefore use the subclass
   **unconditionally** (even when `lam_G == lam_dG`) — it is the only way the
   ridge actually applies.

## Lag / time-alignment convention

`build_augmented_inputs` **drops the first `J` bins of every trial** so all lags
are available without zero-padding (consistent with the existing GLM
convention). All per-trial arrays returned share the same time dimension
`T_n − J`, as ssm requires. ssm's AR(1) aligns the input at index `t` with the
state produced at `t` (`s_t = F s_{t-1} + B tilde_u_t + m_s`); relative to the
spec's `s_{t+1} = F s_t + …` this is a one-bin relabeling that leaves
`G_ℓ, δG_ℓ` unchanged.

## Usage

```python
from perturbation_plds import (
    fit_perturbation_plds, fit_perturbation_plds_no_dG,
    fit_perturbation_plds_intrinsic, evaluate_perturbation_plds,
    crossval_compare_models,
)

# z_list/x_list/u_list: lists (one per trial) of (T_n, 6), (T_n, 90), (T_n,)
res = fit_perturbation_plds(z_list, x_list, u_list, p=10, J=3,
                            lam_G=1e-3, lam_dG=1e-3)
print(res["dG_lag"].shape)   # (J+1, p, d_z) — the perturbation gain

metrics = evaluate_perturbation_plds(res, z_list, x_list, u_list)

# Scientific test: does δG improve held-out fit? (folds split by TRIAL)
cv = crossval_compare_models(z_list, x_list, u_list, p=10, J=3, n_folds=5,
                             models=("full", "no_dG", "intrinsic"))
for m in cv:
    print(m, cv[m]["mean"]["deviance_x_per_obs"])   # lower is better
```

### Results dict keys

`F`, `Q`, `m_s`, `G_lag` (J+1, p, d_z), `dG_lag` (J+1, p, d_z), `C` (d_x, p),
`b` (d_x,), `p`, `J`, `dt`, `log_marginal_likelihood_trace`, `_ssm_model`.
For ablations, absent blocks are zero arrays of the correct shape.

## Files

| File | Contents |
|------|----------|
| `inputs.py` | `build_augmented_inputs`, `build_baseline_inputs` (preprocessing) |
| `model.py`  | `PerBlockRidgeGaussianDynamics`, `NoInputPoissonEmissions` |
| `fit.py`    | `fit_perturbation_plds` and the two ablation fitters |
| `evaluate.py` | `evaluate_perturbation_plds` (Poisson deviance, pseudo-R²) |
| `crossval.py` | trial-level CV (`crossval_perturbation_plds`, `crossval_compare_models`) |
| `synthetic.py`| `generate_synthetic_data`, `run_smoke_test` |

## Inference

Fitting uses ssm's **Laplace-EM** (`method="laplace_em"`) with the structured
mean-field posterior; for an LDS this is exact Kalman smoothing in the
linear-Gaussian limit. We do **not** re-implement filtering/smoothing/EM — all
inference is inherited from ssm. Initialization uses ssm's built-in PCA-on-
`log(counts)` for `C, b`.

## Smoke test

`python -m perturbation_plds.synthetic` generates data with known parameters and
checks: (1) ELBO monotone non-decreasing, (2) recovered `|eig(F)|` close to
truth, (3) smoothed-vs-true latent affine R² > 0.7, (4) model selection — the
full model wins on held-out deviance when `δG_true ≠ 0` and does **not** beat
the no-δG model when `δG_true = 0`.
