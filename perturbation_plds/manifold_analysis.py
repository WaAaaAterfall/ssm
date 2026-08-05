"""Dimensionality / off-manifold analysis for the perturbation PLDS.

Goal (scientific): the calcium + ridge analysis found that unperturbed neural
trajectories live on a low-dimensional plane and perturbed trajectories leave
it, with a larger off-plane angle at every time step.  Here we reproduce and
*strengthen* that claim on the fitted PLDS, whose latent ``s_t`` is the analog of
"calcium after PCA" and whose ``dG`` is the only channel through which the
perturbation can push the latent into new directions.

Three families of primitives:

A. ``off_manifold_angle`` -- time-resolved escape from the unperturbed manifold
   (direct replica of the calcium angle figure).
B. ``participation_ratio`` / ``cumulative_variance`` -- effective dimensionality
   ("perturbed needs a higher-dimensional space").
C. ``gain_subspace_angles`` (parameter-level) and ``counterfactual_latents``
   (causal, holding kinematics fixed) -- the mechanism, which calcium/ridge
   could not provide.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stack(latents):
    """Concatenate a list of (T_n, p) arrays into one (sum T_n, p) matrix."""
    return np.concatenate([np.asarray(s) for s in latents], axis=0)


def fit_manifold(latents, k=None, var_frac=0.90):
    """PCA subspace of a set of latent trajectories.

    Parameters
    ----------
    latents : list of (T_n, p) arrays
        Trajectories that define the reference manifold (e.g. unperturbed).
    k : int, optional
        Number of components to keep.  If ``None``, choose the smallest ``k``
        whose cumulative variance reaches ``var_frac``.
    var_frac : float
        Variance target used when ``k is None``.

    Returns
    -------
    dict with ``mean`` (p,), ``basis`` (p, k) orthonormal columns,
    ``eigvals`` (p,) descending, ``k``.
    """
    X = _stack(latents)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD of centered data; eigenvalues of covariance = s**2 / (n-1)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (s ** 2) / max(Xc.shape[0] - 1, 1)
    if k is None:
        cum = np.cumsum(eigvals) / np.sum(eigvals)
        k = int(np.searchsorted(cum, var_frac) + 1)
    k = max(1, min(k, Vt.shape[0]))
    basis = Vt[:k].T  # (p, k), orthonormal columns
    return dict(mean=mean, basis=basis, eigvals=eigvals, k=k)


# --------------------------------------------------------------------------- #
# A. off-manifold angle
# --------------------------------------------------------------------------- #
def off_manifold_angle(latents, manifold, eps=1e-9):
    """Per-time-step angle of each trajectory off the reference manifold.

    For each state ``s_t`` (centered by the manifold mean), the angle is
    ``arcsin(||s_perp|| / ||s_centered||)`` where ``s_perp`` is the component
    orthogonal to the manifold basis.  0 = on-manifold, pi/2 = fully orthogonal.

    Returns a list of (T_n,) arrays of angles in radians.
    """
    mean, basis = manifold["mean"], manifold["basis"]
    P = basis @ basis.T  # projector onto the manifold
    out = []
    for s in latents:
        sc = np.asarray(s) - mean
        s_par = sc @ P.T
        s_perp = sc - s_par
        norm = np.linalg.norm(sc, axis=1)
        ratio = np.linalg.norm(s_perp, axis=1) / np.clip(norm, eps, None)
        out.append(np.arcsin(np.clip(ratio, 0.0, 1.0)))
    return out


def direction_off_manifold_angle(vectors, basis, eps=1e-9):
    """Angle of each row *direction* off the manifold subspace (NO centering).

    Port of the old calcium "Off-plane angle" plot to the PLDS.  For an input
    drive vector ``v`` (a direction, not a state) the angle off the manifold is
    ``arcsin(||v_perp|| / ||v||)`` where ``v_perp = v - basis (basis^T v)`` is the
    part of ``v`` in the manifold's orthogonal complement.

    This equals the old ``90 - degrees(arccos(|cos(v, plane_normal)|))`` whenever
    the manifold is codimension 1 (a literal plane with a single normal): then
    ``||v_perp|| / ||v|| = |cos(v, normal)|`` and ``arcsin(|cos|) = 90 - arccos(|cos|)``.
    For a k-D manifold the orthogonal complement is (p-k)-dimensional, so this is
    the natural generalization (the normal is replaced by the whole complement).

    Unlike :func:`off_manifold_angle`, the manifold mean is NOT subtracted -- the
    drive is a direction through the origin, exactly as the old code dotted the
    *normalized* drive with the plane normal and ignored the plane offset.

    ``vectors`` : (T, p).  Returns (T,) angles in radians.
    """
    basis = np.asarray(basis)
    P = basis @ basis.T
    v = np.asarray(vectors, dtype=float)
    v_perp = v - v @ P.T
    norm = np.linalg.norm(v, axis=1)
    ratio = np.linalg.norm(v_perp, axis=1) / np.clip(norm, eps, None)
    return np.arcsin(np.clip(ratio, 0.0, 1.0))


def off_manifold_fraction(latents, manifold, eps=1e-9):
    """Per-time-step fraction of (centered) energy off the manifold.

    ``||s_perp||**2 / ||s_centered||**2`` -- a scale-free [0, 1] companion to the
    angle.  Returns a list of (T_n,) arrays.
    """
    mean, basis = manifold["mean"], manifold["basis"]
    P = basis @ basis.T
    out = []
    for s in latents:
        sc = np.asarray(s) - mean
        s_perp = sc - sc @ P.T
        num = np.sum(s_perp ** 2, axis=1)
        den = np.clip(np.sum(sc ** 2, axis=1), eps, None)
        out.append(num / den)
    return out


# --------------------------------------------------------------------------- #
# B. effective dimensionality
# --------------------------------------------------------------------------- #
def participation_ratio(latents, center=True):
    """Participation ratio (effective dimension) of a set of trajectories.

    ``PR = (sum eigvals)**2 / sum(eigvals**2)`` of the latent covariance.
    Scale-invariant: a larger response is not mistaken for a higher-dimensional
    one.  Ranges in [1, p].
    """
    X = _stack(latents)
    if center:
        X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    s1, s2 = ev.sum(), np.sum(ev ** 2)
    return float(s1 ** 2 / s2) if s2 > 0 else 0.0


def participation_ratio_bootstrap(latents, n_boot=500, n_trials=None, seed=0):
    """Bootstrap PR over trials (resample trials with replacement).

    ``n_trials`` subsamples to a fixed trial count so two conditions can be
    compared at equal N (PR estimates depend on sample count).  Returns
    ``(mean, lo, hi)`` with a 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    latents = list(latents)
    n = len(latents)
    m = n if n_trials is None else min(n_trials, n)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=m)
        vals[b] = participation_ratio([latents[i] for i in idx])
    return float(vals.mean()), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def cumulative_variance(latents):
    """Cumulative explained-variance curve (sorted, normalized to 1)."""
    X = _stack(latents)
    X = X - X.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    ev = np.clip(ev, 0, None)
    return np.cumsum(ev) / np.sum(ev)


def offmanifold_variance_fraction(test_latents, manifold):
    """Fraction of test variance NOT captured by the reference manifold.

    Cross-condition leakage: fit the manifold on (held-out) unperturbed trials,
    then measure how much perturbed-trial variance escapes its top-k subspace.
    """
    mean, basis = manifold["mean"], manifold["basis"]
    X = _stack(test_latents) - mean
    total = np.sum(X ** 2)
    proj = X @ basis  # (n, k)
    captured = np.sum(proj ** 2)
    return float(1.0 - captured / total) if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# C. mechanism: parameter subspaces and counterfactual dynamics
# --------------------------------------------------------------------------- #
def _gain_columns(lag):
    """Stack a (J+1, p, d_z) gain tensor into (p, (J+1)*d_z) columns."""
    return np.concatenate([lag[l] for l in range(lag.shape[0])], axis=1)


def gain_subspace_angles(G_lag, dG_lag, tol=1e-8):
    """Compare where baseline kinematics vs perturbation drive the latent.

    Returns a dict with:
      * ``rank_G``, ``rank_dG``, ``rank_joint`` -- numerical ranks of the
        column spaces of G, dG, and [G | dG].  ``rank_joint > rank_G`` means
        the perturbation injects drive into new latent directions.
      * ``principal_angles`` -- principal angles (radians) between col(G) and
        col(dG); angles near pi/2 mean orthogonal (fully new) directions.
      * ``dG_offG_fraction`` -- fraction of dG's drive energy lying outside
        col(G) (1 = entirely new directions).
    """
    from scipy.linalg import subspace_angles, orth

    Gc, dGc = _gain_columns(G_lag), _gain_columns(dG_lag)
    QG = orth(Gc)          # orthonormal basis for col(G)
    rank_G = QG.shape[1]
    rank_dG = orth(dGc).shape[1] if np.linalg.norm(dGc) > tol else 0
    rank_joint = orth(np.concatenate([Gc, dGc], axis=1)).shape[1]

    if rank_G > 0 and rank_dG > 0:
        angles = subspace_angles(Gc, dGc)
    else:
        angles = np.array([])

    # energy of dG outside col(G)
    resid = dGc - QG @ (QG.T @ dGc) if rank_G > 0 else dGc
    denom = np.sum(dGc ** 2)
    off_frac = float(np.sum(resid ** 2) / denom) if denom > 0 else 0.0

    return dict(rank_G=int(rank_G), rank_dG=int(rank_dG),
                rank_joint=int(rank_joint),
                principal_angles=np.asarray(angles),
                dG_offG_fraction=off_frac)


def realized_drive(G_lag, dG_lag, z_full, u_full, which="dG"):
    """Instantaneous input drive into the latent at each trimmed time step.

    Unlike the raw column space of the gains (which, when p <= (J+1)*d_z,
    generically fills R^p and is therefore uninformative), this weights each
    gain by the *actual* kinematics on a trial, so it reveals where the
    perturbation really pushes the latent.

    ``which='dG'`` -> sum_l u_{t-l} dG_l z_{t-l} (perturbation-only drive);
    ``which='G'``  -> sum_l G_l z_{t-l} (baseline drive).  Returns ``(T, p)``.
    """
    z = np.asarray(z_full, dtype=float)
    u = np.asarray(u_full, dtype=float).reshape(-1)
    Jp1, p, d_z = dG_lag.shape
    J = Jp1 - 1
    T = z.shape[0] - J
    out = np.zeros((T, p))
    for t in range(T):
        for l in range(Jp1):
            idx = J + t - l
            if which == "dG":
                out[t] += u[idx] * (dG_lag[l] @ z[idx])
            else:
                out[t] += G_lag[l] @ z[idx]
    return out


def counterfactual_latents(F, G_lag, dG_lag, m_s, z_full, u_full, s0):
    """Roll the *mean* latent dynamics forward with the perturbation on vs off.

    Both rollouts share the same kinematics, the same initial state ``s0``, and
    zero process noise; they differ only in whether the gated dG drive is
    active.  ``delta = s_on - s_off`` is the pure causal effect of the
    perturbation on the latent, holding behavior fixed -- something the calcium
    ridge regression cannot isolate.

    ``z_full`` / ``u_full`` are the *untrimmed* per-trial kinematics/indicator
    ``(T_full, d_z)`` / ``(T_full,)``; the rollout runs over the trimmed
    timeline ``T = T_full - J`` (matching the inferred latents), with the input
    at trimmed step ``t`` built from original index ``J + t - l`` exactly as the
    model was fed.

    Returns ``s_on, s_off, delta`` each ``(T, p)``.
    """
    F = np.asarray(F); m_s = np.asarray(m_s)
    z = np.asarray(z_full, dtype=float)
    u = np.asarray(u_full, dtype=float).reshape(-1)
    Jp1, p, d_z = dG_lag.shape
    J = Jp1 - 1
    T = z.shape[0] - J

    def drive(t, gated):
        d = np.zeros(p)
        for l in range(Jp1):
            idx = J + t - l
            d = d + G_lag[l] @ z[idx]
            if gated:
                d = d + u[idx] * (dG_lag[l] @ z[idx])
        return d

    s_on = np.zeros((T, p)); s_off = np.zeros((T, p))
    s_on[0] = s_off[0] = np.asarray(s0)
    for t in range(1, T):
        s_on[t] = F @ s_on[t - 1] + drive(t, gated=True) + m_s
        s_off[t] = F @ s_off[t - 1] + drive(t, gated=False) + m_s
    return s_on, s_off, (s_on - s_off)


# --------------------------------------------------------------------------- #
# utilities for splitting / aligning trials
# --------------------------------------------------------------------------- #
def perturbation_onset(u):
    """First bin index where u==1, or None if the trial is unperturbed."""
    u = np.asarray(u).reshape(-1)
    nz = np.nonzero(u > 0)[0]
    return int(nz[0]) if nz.size else None


def split_perturbed(u_list):
    """Indices of perturbed vs unperturbed trials given per-trial u arrays."""
    pert, unpert = [], []
    for i, u in enumerate(u_list):
        (pert if np.any(np.asarray(u) > 0) else unpert).append(i)
    return pert, unpert
