"""Evaluation for the perturbation PLDS (Task 3).

Runs the (Kalman/Laplace) smoother to get E[s_t], forms one-step-ahead predicted
firing rates, and reports Poisson deviance / log-likelihood / pseudo-R^2 plus the
held-out (approximate) marginal likelihood.  The deviance formula matches the
existing GLM evaluator.
"""

import numpy as np
from scipy.special import gammaln

from .inputs import build_augmented_inputs, build_baseline_inputs


def _poisson_deviance(y_true, mu, eps=1e-9):
    """Poisson deviance, matching the user's GLM code."""
    mu = np.clip(mu, eps, None)
    yl = np.where(y_true > 0, y_true * np.log((y_true + eps) / mu), 0.0)
    return 2.0 * np.sum(yl - (y_true - mu))


def _poisson_loglik(y_true, mu, eps=1e-9):
    mu = np.clip(mu, eps, None)
    return np.sum(y_true * np.log(mu) - mu - gammaln(y_true + 1.0))


def _build_eval_inputs(z_list, u_list, x_list, results):
    """Rebuild trimmed inputs/counts matching the model the results came from."""
    J = results["J"]
    meta = results["_meta"]
    if not meta["has_G"]:
        # intrinsic model: M = 0, just trim/align via the baseline builder
        _u, x_trim, z_trim, _m = build_baseline_inputs(z_list, u_list, J,
                                                        x_list=x_list)
        return None, x_trim
    if meta["has_dG"]:
        u_aug, x_trim, _z, _m = build_augmented_inputs(z_list, u_list, J,
                                                       x_list=x_list)
    else:
        u_aug, x_trim, _z, _m = build_baseline_inputs(z_list, u_list, J,
                                                      x_list=x_list)
    return u_aug, x_trim


def evaluate_perturbation_plds(results, z_list, x_list, u_list):
    """Evaluate a fitted perturbation PLDS on (possibly held-out) trials.

    Returns
    -------
    dict with keys: ``deviance_x``, ``deviance_x_per_obs``, ``loglik_x``,
    ``pseudoR2_x`` (vs a constant per-neuron rate null), and
    ``held_out_marginal_likelihood`` (the Laplace-EM ELBO of the smoother, which
    for the Gaussian-dynamics Poisson LDS is the standard variational lower bound
    on log p(x); exact only in the linear-Gaussian limit).
    """
    lds = results["_ssm_model"]
    F = results["F"]
    m_s = results["m_s"]
    C = results["C"]
    b = results["b"]
    dt = results["dt"]
    M = results["_meta"]["M"]
    B = np.array(lds.dynamics.Vs[0]) if M > 0 else None

    u_aug, x_trim = _build_eval_inputs(z_list, u_list, x_list, results)

    # Run the smoother with fixed parameters (no learning).
    elbos, posterior = lds.approximate_posterior(
        x_trim, inputs=(u_aug if M > 0 else None),
        method="laplace_em", verbose=0)
    Es_list = posterior.mean_continuous_states  # list of (T_n, p)

    all_y, all_mu = [], []
    for n, (Es, x) in enumerate(zip(Es_list, x_trim)):
        T = Es.shape[0]
        if T < 2:
            continue
        # one-step-ahead predicted state: s_pred[t] = F Es[t-1] + B u[t] + m_s
        s_pred = Es[:-1] @ F.T + m_s  # (T-1, p), predicting t = 1..T-1
        if M > 0:
            s_pred = s_pred + u_aug[n][1:] @ B.T
        log_rate = s_pred @ C.T + b   # (T-1, d_x)
        mu = dt * np.exp(log_rate)
        all_y.append(x[1:])
        all_mu.append(mu)

    y_true = np.concatenate(all_y, axis=0)
    mu = np.concatenate(all_mu, axis=0)
    n_obs = y_true.size

    deviance_x = _poisson_deviance(y_true, mu)
    loglik_x = _poisson_loglik(y_true, mu)

    # Constant-rate null: per-neuron mean count over the evaluated observations.
    null_rate = np.mean(y_true, axis=0, keepdims=True)  # (1, d_x)
    null_mu = np.broadcast_to(null_rate, y_true.shape)
    deviance_null = _poisson_deviance(y_true, null_mu)
    pseudoR2_x = 1.0 - deviance_x / deviance_null if deviance_null > 0 else 0.0

    return dict(
        deviance_x=float(deviance_x),
        deviance_x_per_obs=float(deviance_x / n_obs),
        loglik_x=float(loglik_x),
        pseudoR2_x=float(pseudoR2_x),
        held_out_marginal_likelihood=float(elbos[-1]),
        n_obs=int(n_obs),
    )
