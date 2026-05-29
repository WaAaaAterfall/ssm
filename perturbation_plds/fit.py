"""Fitting routines for the perturbation PLDS (Tasks 2 and 5).

The full model and both nested ablations all reduce to: build a deterministic
augmented input, fit an off-the-shelf input-driven Poisson ``ssm.LDS`` (with our
``PerBlockRidgeGaussianDynamics`` and ``NoInputPoissonEmissions``), then unpack
the fitted parameters into the scientifically meaningful blocks.
"""

import numpy as np

from ssm.lds import LDS

from .inputs import build_augmented_inputs, build_baseline_inputs
from .model import PerBlockRidgeGaussianDynamics, NoInputPoissonEmissions

DEFAULT_DT = 5.0 / 240.0


def _build_lds(d_x, p, M, dt, lam_F, lam_G, lam_dG, block_split):
    """Construct an LDS with frozen-input Poisson emissions and ridge dynamics."""
    dynamics = PerBlockRidgeGaussianDynamics(
        1, p, M=M, lags=1,
        lam_F=lam_F, lam_G=lam_G, lam_dG=lam_dG, block_split=block_split)
    emissions = NoInputPoissonEmissions(
        d_x, 1, p, M=M, single_subspace=True, link="log", bin_size=dt)
    return LDS(d_x, p, M=M, dynamics=dynamics, emissions=emissions)


def _unpack_input_matrix(B, J, d_z, block_split):
    """Split B (p x M) into G_lag (J+1, p, d_z) and dG_lag (J+1, p, d_z)."""
    p = B.shape[0]
    G_lag = np.zeros((J + 1, p, d_z))
    dG_lag = np.zeros((J + 1, p, d_z))
    for l in range(J + 1):
        G_lag[l] = B[:, l * d_z:(l + 1) * d_z]
        if block_split + (l + 1) * d_z <= B.shape[1]:
            dG_lag[l] = B[:, block_split + l * d_z: block_split + (l + 1) * d_z]
    return G_lag, dG_lag


def _extract_results(lds, elbos, *, p, J, d_z, dt, block_split, has_dG, has_G):
    """Pull F, Q, m_s, G/dG, C, b out of a fitted LDS into a results dict."""
    dyn = lds.dynamics
    emi = lds.emissions

    F = np.array(dyn.As[0])           # (p, p)
    Q = np.array(dyn.Sigmas[0])       # (p, p)
    m_s = np.array(dyn.bs[0])         # (p,)
    B = np.array(dyn.Vs[0])           # (p, M)

    if has_G:
        G_lag, dG_lag = _unpack_input_matrix(B, J, d_z, block_split)
        if not has_dG:
            dG_lag = np.zeros((J + 1, p, d_z))
    else:
        G_lag = np.zeros((J + 1, p, d_z))
        dG_lag = np.zeros((J + 1, p, d_z))

    C = np.array(emi.Cs[0])           # (d_x, p)
    b = np.array(emi.ds[0])           # (d_x,)

    return dict(
        F=F, Q=Q, m_s=m_s,
        G_lag=G_lag, dG_lag=dG_lag,
        C=C, b=b,
        p=p, J=J, dt=dt,
        log_marginal_likelihood_trace=np.asarray(elbos),
        _ssm_model=lds,
        _meta=dict(d_z=d_z, M=B.shape[1], block_split=block_split,
                   has_G=has_G, has_dG=has_dG),
    )


def _fit_core(u_aug_list, x_trim_list, meta, *, p, J, dt, n_iters, method,
              lam_F, lam_G, lam_dG, has_dG, has_G, verbose, num_init_iters):
    """Shared fitting body given prebuilt inputs."""
    d_x = x_trim_list[0].shape[1]
    d_z = meta["d_z"]
    M = meta["M"]
    block_split = meta["block_split"]

    lds = _build_lds(d_x, p, M, dt, lam_F, lam_G, lam_dG, block_split)
    elbos, _posterior = lds.fit(
        x_trim_list, inputs=(u_aug_list if M > 0 else None),
        method=method, num_iters=n_iters,
        initialize=True, num_init_iters=num_init_iters,
        verbose=verbose)

    return _extract_results(lds, elbos, p=p, J=J, d_z=d_z, dt=dt,
                            block_split=block_split, has_dG=has_dG, has_G=has_G)


def fit_perturbation_plds(z_list, x_list, u_list, *, p, J, dt=DEFAULT_DT,
                          n_iters=100, method="laplace_em",
                          lam_G=1e-4, lam_dG=1e-4, lam_F=0.0,
                          verbose=2, num_init_iters=25):
    """Fit the full perturbation PLDS.

    Parameters
    ----------
    z_list, x_list, u_list : lists of per-trial arrays
        Kinematics ``(T_n, d_z)``, spike counts ``(T_n, d_x)``, perturbation
        indicator ``(T_n,)``.
    p : int
        Latent dimension.
    J : int
        Kinematic lag order.
    dt : float
        Bin width (seconds); used as the Poisson ``bin_size``.
    lam_G, lam_dG, lam_F : float
        Ridge penalties on the G block, dG block, and F respectively.  For the
        basic fit set ``lam_G == lam_dG``; Task 4's per-block behavior is active
        whenever they differ.

    Returns
    -------
    dict with keys 'F', 'Q', 'm_s', 'G_lag' (J+1, p, d_z),
    'dG_lag' (J+1, p, d_z), 'C' (d_x, p), 'b' (d_x,), 'p', 'J', 'dt',
    'log_marginal_likelihood_trace', and '_ssm_model'.
    """
    u_aug_list, x_trim_list, _z_trim, meta = build_augmented_inputs(
        z_list, u_list, J, x_list=x_list)
    return _fit_core(u_aug_list, x_trim_list, meta, p=p, J=J, dt=dt,
                     n_iters=n_iters, method=method,
                     lam_F=lam_F, lam_G=lam_G, lam_dG=lam_dG,
                     has_dG=True, has_G=True,
                     verbose=verbose, num_init_iters=num_init_iters)


def fit_perturbation_plds_no_dG(z_list, x_list, u_list, *, p, J, dt=DEFAULT_DT,
                                n_iters=100, method="laplace_em",
                                lam_G=1e-4, lam_dG=1e-4, lam_F=0.0,
                                verbose=2, num_init_iters=25):
    """Ablation with dG_l = 0 fixed (kinematics drive the latent, no gating).

    Implemented by building the augmented input WITHOUT the gated columns.
    ``lam_dG`` is accepted for signature compatibility but unused.  ``dG_lag``
    in the returned dict is a zero array of shape ``(J+1, p, d_z)``.
    """
    u_aug_list, x_trim_list, _z_trim, meta = build_baseline_inputs(
        z_list, u_list, J, x_list=x_list)
    return _fit_core(u_aug_list, x_trim_list, meta, p=p, J=J, dt=dt,
                     n_iters=n_iters, method=method,
                     lam_F=lam_F, lam_G=lam_G, lam_dG=lam_dG,
                     has_dG=False, has_G=True,
                     verbose=verbose, num_init_iters=num_init_iters)


def fit_perturbation_plds_intrinsic(z_list, x_list, u_list, *, p, J,
                                    dt=DEFAULT_DT, n_iters=100,
                                    method="laplace_em",
                                    lam_F=0.0, verbose=2, num_init_iters=25,
                                    **_ignored):
    """Autonomous PLDS: no kinematic inputs at all (M = 0).

    The first ``J`` bins are still dropped so the held-out comparison against the
    other two models uses identical observations.  Both ``G_lag`` and ``dG_lag``
    are returned as zero arrays.
    """
    # Reuse the baseline builder only to trim/align x and z consistently.
    _u, x_trim_list, _z_trim, base_meta = build_baseline_inputs(
        z_list, u_list, J, x_list=x_list)
    meta = dict(J=J, d_z=base_meta["d_z"], block_split=0, M=0)
    return _fit_core(None, x_trim_list, meta, p=p, J=J, dt=dt,
                     n_iters=n_iters, method=method,
                     lam_F=lam_F, lam_G=0.0, lam_dG=0.0,
                     has_dG=False, has_G=False,
                     verbose=verbose, num_init_iters=num_init_iters)
