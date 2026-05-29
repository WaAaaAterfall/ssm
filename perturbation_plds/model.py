"""Customized ssm components for the perturbation PLDS.

Two thin subclasses live here.  Neither modifies the ``ssm`` library; both are
ordinary subclasses defined in this package.

1. ``PerBlockRidgeGaussianDynamics`` -- the one *real* customization (Task 4).
   It applies different ridge penalties to different column blocks of the input
   matrix B = [G_0..G_J | dG_0..dG_J], plus an optional penalty on F.

2. ``NoInputPoissonEmissions`` -- holds the emission input matrix ``Fs`` at zero
   so that observations depend *only* on the latent state, exactly as the model
   spec requires (x_t ~ Poisson(dt * exp(C s_t + b))).  ssm's ``LDS`` shares a
   single input dimension ``M`` across dynamics and emissions, so off-the-shelf
   the emissions would learn a direct kinematics->spikes path that bypasses the
   latent state.  Freezing ``Fs = 0`` removes that path.
"""

import autograd.numpy as np

from ssm.observations import AutoRegressiveObservations
from ssm.emissions import PoissonEmissions
from ssm.util import ensure_args_are_lists


class PerBlockRidgeGaussianDynamics(AutoRegressiveObservations):
    """Gaussian AR(1) dynamics with per-block ridge regularization.

    ssm's AR M-step solves the regularized normal equations

        W = (sum_t E[r_t r_t^T] + J0)^{-1} (sum_t E[s_{t+1} r_t^T] + h0)^T

    where ``r_t = [s_t ; tilde_u_t ; 1]`` and ``J0`` is a diagonal prior
    precision over the columns ``[F-block | B-block | bias]``.  This is exactly
    the per-block ridge requested: we set ``J0`` to ``lam_F`` on the F columns,
    ``lam_G`` on the baseline-kinematic (G) columns, ``lam_dG`` on the gated
    (dG) columns, and 0 on the bias, with ``h0 = 0`` (shrinkage toward zero).

    Implementation note
    -------------------
    For a plain ``AutoRegressiveObservations`` with ``lags == 1`` ssm runs an
    *exact* M-step that ignores ``J0``/``h0`` entirely (so a ridge penalty would
    silently do nothing).  Because ssm selects that path with an exact
    ``type(...) in [...]`` check, this subclass is *not* matched and instead
    falls back to the prior-respecting sufficient-statistics M-step.  We use this
    class unconditionally (even when ``lam_G == lam_dG``) so the penalty applies.
    """

    def __init__(self, K, D, M=0, lags=1,
                 lam_F=0.0, lam_G=1e-4, lam_dG=1e-4, block_split=None,
                 nu0=1e-4, Psi0=1e-4):
        super(PerBlockRidgeGaussianDynamics, self).__init__(
            K, D, M=M, lags=lags, nu0=nu0, Psi0=Psi0)
        assert lags == 1, "perturbation PLDS uses AR(1) dynamics"
        self.lam_F = float(lam_F)
        self.lam_G = float(lam_G)
        self.lam_dG = float(lam_dG)
        # number of baseline (G) input columns; remainder are gated (dG) columns
        self.block_split = M if block_split is None else int(block_split)
        assert 0 <= self.block_split <= M
        self._set_block_ridge()

    def _set_block_ridge(self):
        """(Re)build the diagonal prior precision J0 and prior mean term h0."""
        K, D, M, lags = self.K, self.D, self.M, self.lags
        D_in = D * lags + M + 1  # [F-block | B-block | bias]

        diag = np.zeros(D_in)
        diag[:D * lags] = self.lam_F
        # baseline-kinematic (G) columns
        diag[D * lags:D * lags + self.block_split] = self.lam_G
        # gated (dG) columns
        diag[D * lags + self.block_split:D * lags + M] = self.lam_dG
        # bias column (index -1) stays 0 -> unpenalized

        self.J0 = np.tile(np.diag(diag)[None, :, :], (K, 1, 1))
        # h0 = 0  =>  shrink every penalized column toward zero (pure ridge).
        self.h0 = np.zeros((K, D_in, D))

    def m_step(self, expectations, datas, inputs, masks, tags,
               continuous_expectations=None, **kwargs):
        # Refresh the penalty in case it was changed after construction, and
        # force the prior-respecting M-step (drop continuous_expectations, whose
        # presence would trigger ssm's prior-free exact M-step).
        self._set_block_ridge()
        super(PerBlockRidgeGaussianDynamics, self).m_step(
            expectations, datas, inputs, masks, tags,
            continuous_expectations=None, **kwargs)


class NoInputPoissonEmissions(PoissonEmissions):
    """Poisson emissions whose input matrix ``Fs`` is held at zero.

    The latent-to-observation map is the full Poisson GLM ``dt * exp(C s + b)``
    with no direct input term.  ``Fs`` is excluded from ``params`` so the
    emission M-step never updates it, and it is re-zeroed after initialization.
    """

    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(NoInputPoissonEmissions, self).__init__(
            N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.Fs = np.zeros_like(self.Fs)

    @property
    def params(self):
        # Exclude Fs: it is frozen at zero.
        return self.Cs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.ds = self.ds[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize C, b via PCA on log(counts) with the input contribution
        # removed -- but since Fs is frozen at zero, run the regression-based
        # init with zeroed inputs so the residual is the full data.
        zero_inputs = [np.zeros_like(u) for u in inputs]
        super(NoInputPoissonEmissions, self).initialize(
            datas, inputs=zero_inputs, masks=masks, tags=tags)
        self.Fs = np.zeros_like(self.Fs)
