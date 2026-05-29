"""Preprocessing for the perturbation PLDS.

The core "customization" of this model is *not* new model code -- it is the
construction of an augmented, deterministic input time series that the standard
input-driven LDS in ``ssm`` treats as just another set of input columns.

Given kinematics ``z_t`` (d_z = 6) and a known binary perturbation indicator
``u_t`` we build, for each bin t,

    tilde_u_t = [ z_t, z_{t-1}, ..., z_{t-J},                # baseline kinematics
                  u_t * z_t, u_{t-1} * z_{t-1}, ..., u_{t-J} * z_{t-J} ]  # gated

so that ``tilde_u_t`` has dimension ``2 * (J + 1) * d_z``.  When this is fed to
an input-driven LDS, the library's input matrix ``B`` (``dynamics.Vs[0]``)
represents the concatenation ``[G_0, ..., G_J, dG_0, ..., dG_J]``, which we
unpack after fitting.
"""

import numpy as np


def build_augmented_inputs(z_list, u_list, J, x_list=None):
    """Construct augmented input arrays (lagged + gated-lagged kinematics).

    Lag convention
    --------------
    We drop the first ``J`` bins of every trial (consistent with the existing
    GLM convention), so that every lag ``z_{t-l}`` for ``l = 0..J`` is available
    without zero padding.  All per-trial arrays returned therefore have the same
    time dimension ``T_n - J`` within a trial, which is what ``ssm`` requires.

    Note on alignment with the written state equation: the spec writes
    ``s_{t+1} = F s_t + sum_l G_l z_{t-l} + ...``.  ``ssm``'s AR(1) dynamics are
    ``s_t = F s_{t-1} + B tilde_u_t + m_s``, i.e. the input at index ``t`` drives
    the state produced at index ``t``.  Feeding ``tilde_u`` as defined above is
    exactly the input-driven LDS the spec asks for; the bin index is relabelled
    by one but the parameters (G_l, dG_l) are unchanged.

    Parameters
    ----------
    z_list : list of np.ndarray
        Per-trial kinematics, each ``(T_n, d_z)`` float.
    u_list : list of np.ndarray
        Per-trial perturbation indicators, each ``(T_n,)`` in {0, 1}.
    J : int
        Kinematic lag order.
    x_list : list of np.ndarray, optional
        Per-trial spike counts, each ``(T_n, d_x)``.  If given, returned trimmed
        and cast to int (ssm's Poisson likelihood requires integer counts).

    Returns
    -------
    u_aug_list : list of np.ndarray
        Each ``(T_n - J, 2 * (J + 1) * d_z)`` input array.
    x_trim_list : list of np.ndarray or None
        ``x`` truncated to ``[J:]`` (int dtype), or ``None`` if ``x_list`` is None.
    z_trim_list : list of np.ndarray
        ``z`` truncated to ``[J:]`` to match the input time alignment.
    meta : dict
        Keys: ``'J'``, ``'d_z'``, ``'block_split'`` (= ``(J+1)*d_z``, the number
        of baseline-kinematic columns, i.e. the boundary between the G block and
        the dG block), and ``'M'`` (total input dimension ``2*(J+1)*d_z``).
    """
    assert J >= 0
    d_z = z_list[0].shape[1]
    block_split = (J + 1) * d_z
    M = 2 * block_split

    u_aug_list, z_trim_list = [], []
    x_trim_list = [] if x_list is not None else None

    for n, (z, u) in enumerate(zip(z_list, u_list)):
        z = np.asarray(z, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        T = z.shape[0]
        assert z.shape[1] == d_z, "inconsistent d_z across trials"
        assert u.shape[0] == T, "z and u length mismatch in trial %d" % n
        assert T > J, "trial %d too short for J=%d" % (n, J)

        # lag l contributes rows [J-l : T-l], whose i-th row is t-l for t=J+i.
        base = np.concatenate([z[J - l:T - l] for l in range(J + 1)], axis=1)
        gated = np.concatenate(
            [u[J - l:T - l][:, None] * z[J - l:T - l] for l in range(J + 1)],
            axis=1)
        u_aug = np.concatenate([base, gated], axis=1)
        assert u_aug.shape == (T - J, M)

        u_aug_list.append(u_aug)
        z_trim_list.append(z[J:])
        if x_list is not None:
            x = np.asarray(x_list[n])
            assert x.shape[0] == T, "z and x length mismatch in trial %d" % n
            x_trim_list.append(np.round(x[J:]).astype(int))

    meta = dict(J=J, d_z=d_z, block_split=block_split, M=M)
    return u_aug_list, x_trim_list, z_trim_list, meta


def build_baseline_inputs(z_list, u_list, J, x_list=None):
    """Augmented inputs WITHOUT the gated columns (for the no-dG ablation).

    Identical time alignment to :func:`build_augmented_inputs`, but the input is
    just ``[z_t, ..., z_{t-J}]`` with dimension ``(J + 1) * d_z``.  ``u_list`` is
    accepted for a uniform call signature but unused.
    """
    d_z = z_list[0].shape[1]
    block_split = (J + 1) * d_z

    u_aug_list, z_trim_list = [], []
    x_trim_list = [] if x_list is not None else None
    for n, z in enumerate(z_list):
        z = np.asarray(z, dtype=float)
        T = z.shape[0]
        assert T > J, "trial %d too short for J=%d" % (n, J)
        base = np.concatenate([z[J - l:T - l] for l in range(J + 1)], axis=1)
        u_aug_list.append(base)
        z_trim_list.append(z[J:])
        if x_list is not None:
            x_trim_list.append(np.round(np.asarray(x_list[n])[J:]).astype(int))

    meta = dict(J=J, d_z=d_z, block_split=block_split, M=block_split)
    return u_aug_list, x_trim_list, z_trim_list, meta
