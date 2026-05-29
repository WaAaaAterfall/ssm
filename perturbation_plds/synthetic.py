"""Synthetic data and smoke test for the perturbation PLDS (Task 7).

``generate_synthetic_data`` draws from the model with known parameters, choosing
emission baselines so that per-neuron mean counts land in a realistic range (so
the recurrent signal is identifiable).  ``run_smoke_test`` fits the model and
checks the sanity conditions from the spec.
"""

import numpy as np

from .fit import fit_perturbation_plds
from .inputs import build_augmented_inputs
from .crossval import crossval_compare_models

DEFAULT_DT = 5.0 / 240.0


def _stable_matrix(p, rng, radius=0.8):
    A = rng.randn(p, p)
    A = A / np.max(np.abs(np.linalg.eigvals(A)))  # spectral radius 1
    return radius * A


def _generate_kinematics(T, d_z, rng, rho=0.95, scale=1.0):
    """Smooth AR(1) kinematic series (T, d_z)."""
    z = np.zeros((T, d_z))
    for t in range(1, T):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.randn(d_z)
    return scale * z


def generate_synthetic_data(p_true=4, J=2, N_trials=50, T=160, d_x=60, d_z=6,
                            dt=DEFAULT_DT, dG_scale=1.0, G_scale=0.3,
                            mean_count_range=(0.05, 0.5), u_prob=0.3,
                            C_signal_std=0.35, q_std=0.05, seed=0):
    """Sample trials from the perturbation PLDS with known parameters.

    Returns
    -------
    z_list, x_list, u_list : lists of length N_trials
    true : dict with keys 'F', 'G_lag', 'dG_lag', 'm_s', 'Q', 'C', 'b',
           's_list', 'dt', 'p', 'J', 'dG_scale'.
    """
    rng = np.random.RandomState(seed)
    p = p_true

    F = _stable_matrix(p, rng, radius=0.8)
    G_lag = G_scale * rng.randn(J + 1, p, d_z) / np.sqrt(d_z)
    dG_lag = dG_scale * rng.randn(J + 1, p, d_z) / np.sqrt(d_z)
    m_s = 0.02 * rng.randn(p)
    Q = (q_std ** 2) * np.eye(p)
    Q_chol = np.linalg.cholesky(Q)

    # Emission loading; scaled below so that C @ s has std ~ C_signal_std.
    C = rng.randn(d_x, p)

    # --- simulate latents ---
    z_list, u_list, s_list = [], [], []
    for n in range(N_trials):
        z = _generate_kinematics(T, d_z, rng)
        u = (rng.rand(T) < u_prob).astype(float)
        s = np.zeros((T, p))
        s[0] = q_std * rng.randn(p)
        for t in range(1, T):
            drive = m_s.copy()
            for l in range(J + 1):
                if t - l >= 0:
                    drive = drive + G_lag[l] @ z[t - l]
                    drive = drive + u[t - l] * (dG_lag[l] @ z[t - l])
            s[t] = F @ s[t - 1] + drive + Q_chol @ rng.randn(p)
        z_list.append(z)
        u_list.append(u)
        s_list.append(s)

    S = np.concatenate(s_list, axis=0)  # (N*T, p)

    # Scale C so the latent signal entering the log-rate has controlled std.
    proj = S @ C.T  # (NT, d_x)
    cur_std = np.mean(np.std(proj, axis=0))
    C = C * (C_signal_std / (cur_std + 1e-12))
    proj = S @ C.T

    # Choose per-neuron baseline b to hit a target mean count.
    targets = rng.uniform(mean_count_range[0], mean_count_range[1], size=d_x)
    mean_exp = np.mean(np.exp(proj), axis=0)  # E_t[exp(C_i s_t)]
    b = np.log(targets / dt) - np.log(mean_exp)

    # --- sample spikes ---
    x_list = []
    for s in s_list:
        log_rate = s @ C.T + b
        rate = dt * np.exp(log_rate)
        x_list.append(rng.poisson(rate).astype(int))

    true = dict(F=F, G_lag=G_lag, dG_lag=dG_lag, m_s=m_s, Q=Q, C=C, b=b,
                s_list=s_list, dt=dt, p=p, J=J, dG_scale=dG_scale)
    return z_list, x_list, u_list, true


def _latent_r2(Es_list, true_s_list, J):
    """Affinely regress smoothed latents onto true latents; return mean R^2.

    The latent state is identified only up to an invertible affine transform, so
    we fit ``true_s ~ [Es, 1] W`` by least squares and report the R^2.
    """
    X, Y = [], []
    for Es, s_true in zip(Es_list, true_s_list):
        s_true_trim = s_true[J:]  # align with dropped first J bins
        T = min(Es.shape[0], s_true_trim.shape[0])
        X.append(np.column_stack([Es[:T], np.ones(T)]))
        Y.append(s_true_trim[:T])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Yhat = X @ W
    ss_res = np.sum((Y - Yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(np.mean(r2)), r2


def _smoothed_latents(results, z_list, x_list, u_list):
    lds = results["_ssm_model"]
    J = results["J"]
    u_aug, x_trim, _z, _m = build_augmented_inputs(z_list, u_list, J,
                                                   x_list=x_list)
    _elbos, posterior = lds.approximate_posterior(
        x_trim, inputs=u_aug, method="laplace_em", verbose=0)
    return posterior.mean_continuous_states


def run_smoke_test(p_fit=6, J=2, N_trials=50, T=160, d_x=60, d_z=6,
                   n_iters=100, num_init_iters=25, n_folds=2,
                   cv_n_iters=30, cv_num_init_iters=10,
                   run_fit_checks=True, lam=1e-3, seed=0, verbose=False):
    """Run the Task-7 sanity checks and print a report.

    Checks 1-3 use one full fit on the dG!=0 dataset.  Check 4 (model
    selection) cross-validates full vs no_dG on both a dG!=0 and a dG==0
    dataset -- that is ``2 datasets x 2 models x n_folds`` fits, the minimum
    needed to show the comparison works in *both* directions.  Use a small
    ``n_folds`` (a single held-out split per direction is enough for a smoke
    test) and a smaller ``cv_n_iters`` to keep this cheap.  Set
    ``run_fit_checks=False`` to run only the model-selection check.

    Returns a dict of all computed diagnostics.
    """
    rng_seed = seed
    out = {}
    log = (lambda *a: print(*a)) if True else (lambda *a: None)

    log("=" * 70)
    log("PERTURBATION PLDS SMOKE TEST")
    log("=" * 70)

    # ---- dataset with dG != 0 ----
    log("\n[1/2] Generating data with dG_true != 0 ...")
    zB, xB, uB, trueB = generate_synthetic_data(
        p_true=4, J=J, N_trials=N_trials, T=T, d_x=d_x, d_z=d_z,
        dG_scale=1.0, seed=rng_seed)
    mc = np.mean([x.mean() for x in xB])
    log("    mean count / bin / neuron = %.3f (target 0.05-0.5)" % mc)

    monotone, ev_err, r2, r2_per, elbos = True, None, None, None, None
    if run_fit_checks:
        log("    Fitting full model (p_fit=%d, J=%d) ..." % (p_fit, J))
        resB = fit_perturbation_plds(zB, xB, uB, p=p_fit, J=J, n_iters=n_iters,
                                     lam_G=lam, lam_dG=lam,
                                     num_init_iters=num_init_iters,
                                     verbose=2 if verbose else 0)

        elbos = resB["log_marginal_likelihood_trace"]
        diffs = np.diff(elbos)
        # A "violation" is a downward step that is a non-trivial fraction of the
        # total ELBO ascent.  Laplace-EM with partial (alpha<1), sampled
        # M-steps is not guaranteed exactly monotone, so we allow dips up to 2%
        # of total ascent.
        ascent = float(elbos[-1] - elbos[0])
        tol = max(2e-2 * abs(ascent), 1e-6 * (abs(elbos[-1]) + 1.0))
        max_drop = float(-diffs.min()) if len(diffs) else 0.0
        monotone = bool(np.all(diffs > -tol))
        log("\n  CHECK 1 (ELBO monotone non-decreasing): %s" % monotone)
        log("    ELBO start=%.1f end=%.1f  max single-step drop=%.4g (tol=%.4g)"
            % (elbos[0], elbos[-1], max_drop, tol))

        ev_true = np.sort(np.abs(np.linalg.eigvals(trueB["F"])))[::-1]
        ev_fit = np.sort(np.abs(np.linalg.eigvals(resB["F"])))[::-1]
        k = min(len(ev_true), len(ev_fit))
        ev_err = float(np.mean(np.abs(ev_true[:k] - ev_fit[:k])))
        log("\n  CHECK 2 (dominant |eig(F)|):")
        log("    true: %s" % np.array2string(ev_true, precision=3))
        log("    fit : %s" % np.array2string(ev_fit[:k], precision=3))
        log("    mean abs error on top-%d magnitudes = %.3f" % (k, ev_err))

        Es_list = _smoothed_latents(resB, zB, xB, uB)
        r2, r2_per = _latent_r2(Es_list, trueB["s_list"], J)
        log("\n  CHECK 3 (latent affine R^2): %.3f  (threshold ~0.7)" % r2)

    # ---- model selection on both datasets ----
    log("\n[2/2] Cross-validated model selection (n_folds=%d, cv_n_iters=%d) ..."
        % (n_folds, cv_n_iters))
    log("    Dataset B (dG_true != 0): comparing full vs no_dG ...")
    cvB = crossval_compare_models(zB, xB, uB, p=p_fit, J=J, n_folds=n_folds,
                                  lam_G=lam, lam_dG=lam, n_iters=cv_n_iters,
                                  num_init_iters=cv_num_init_iters, seed=seed,
                                  models=("full", "no_dG"))
    devB_full = cvB["full"]["mean"]["deviance_x_per_obs"]
    devB_nodG = cvB["no_dG"]["mean"]["deviance_x_per_obs"]
    log("      held-out deviance/obs:  full=%.5f   no_dG=%.5f" %
        (devB_full, devB_nodG))
    full_wins_B = devB_full < devB_nodG

    log("    Dataset A (dG_true == 0): comparing full vs no_dG ...")
    zA, xA, uA, trueA = generate_synthetic_data(
        p_true=4, J=J, N_trials=N_trials, T=T, d_x=d_x, d_z=d_z,
        dG_scale=0.0, seed=rng_seed + 1)
    cvA = crossval_compare_models(zA, xA, uA, p=p_fit, J=J, n_folds=n_folds,
                                  lam_G=lam, lam_dG=lam, n_iters=cv_n_iters,
                                  num_init_iters=cv_num_init_iters, seed=seed,
                                  models=("full", "no_dG"))
    devA_full = cvA["full"]["mean"]["deviance_x_per_obs"]
    devA_nodG = cvA["no_dG"]["mean"]["deviance_x_per_obs"]
    log("      held-out deviance/obs:  full=%.5f   no_dG=%.5f" %
        (devA_full, devA_nodG))
    # "no better": full should not beat no_dG by a meaningful margin.
    margin = 0.01 * abs(devA_nodG)
    full_not_better_A = devA_full > devA_nodG - margin

    log("\n  CHECK 4 (model selection):")
    log("    dG!=0  -> full should win on held-out: %s" % full_wins_B)
    log("    dG==0  -> full should NOT beat no_dG : %s" % full_not_better_A)

    checks = dict(
        elbo_monotone=monotone,
        eig_error=ev_err,
        latent_r2=r2,
        full_wins_when_dG=full_wins_B,
        full_not_better_when_no_dG=full_not_better_A,
    )
    fit_checks_ok = (not run_fit_checks) or (monotone and r2 is not None and r2 > 0.7)
    passed = (fit_checks_ok and full_wins_B and full_not_better_A)
    log("\n" + "=" * 70)
    log("SMOKE TEST %s" % ("PASSED" if passed else "needs review (see above)"))
    log("=" * 70)

    out.update(checks)
    out["passed"] = bool(passed)
    out["cv_dG"] = cvB
    out["cv_no_dG_data"] = cvA
    out["elbos"] = elbos
    out["latent_r2_per_dim"] = r2_per
    return out


if __name__ == "__main__":
    run_smoke_test()
