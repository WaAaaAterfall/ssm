# Perturbation PLDS — model structure

Mathematical specification of the customized Poisson Linear Dynamical System in
`perturbation_plds/`, in the order the code realizes it.

Preview in VS Code with `Ctrl+Shift+V`.

---

## 1. Notation

| Symbol | Space | Meaning |
|---|---|---|
| $s_t$ | $\mathbb{R}^p$ | latent neural state, $p \in \{5,10,15,20\}$ |
| $z_t$ | $\mathbb{R}^{d_z}$ | kinematics, $d_z = 6$ (3D position + 3D velocity) |
| $x_t$ | $\mathbb{Z}_{\geq 0}^{d_x}$ | binned spike counts, $d_x = 90$ neurons |
| $u_t$ | $\{0,1\}$ | known perturbation indicator |
| $\Delta$ | $\mathbb{R}_{>0}$ | bin width, $\Delta = 5/240$ s |
| $J$ | $\mathbb{N}$ | kinematic lag order, $J \in \{2,3,4\}$ |

Trials $n = 1,\dots,N$ are independent and share all parameters; $t = 0,\dots,T_n-1$.

---

## 2. Generative model

**State equation** — linear-Gaussian, input-driven:

$$
s_{t+1} \;=\; F s_t \;+\; \sum_{\ell=0}^{J} G_\ell\, z_{t-\ell} \;+\; \sum_{\ell=0}^{J} u_{t-\ell}\, \delta G_\ell\, z_{t-\ell} \;+\; m_s \;+\; \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0, Q)
$$

**Observation equation** — Poisson with log link, conditionally independent across neurons:

$$
x_t^{(i)} \mid s_t \;\sim\; \operatorname{Poisson}\!\Big( \Delta \cdot \exp\big( c_i^\top s_t + b_i \big) \Big),
\qquad i = 1,\dots,d_x
$$

**Parameters**

$$
F \in \mathbb{R}^{p \times p}, \quad
Q \in \mathbb{S}_{++}^{p}, \quad
m_s \in \mathbb{R}^{p}, \quad
G_\ell,\, \delta G_\ell \in \mathbb{R}^{p \times d_z}\ (\ell = 0,\dots,J),
$$

$$
C = [\,c_1\ \cdots\ c_{d_x}\,]^\top \in \mathbb{R}^{d_x \times p}, \quad
b \in \mathbb{R}^{d_x}
$$

$\delta G_\ell$ — how the perturbation changes the kinematics $\to$ latent drive — is the
scientific target. Everything else exists to estimate it and test whether it is real.

---

## 3. Reduction to a standard input-driven LDS

Both sums are linear in **known, deterministic** quantities. Define the augmented input

$$
\tilde{u}_t \;=\;
\big[\, z_t^\top,\ z_{t-1}^\top,\ \dots,\ z_{t-J}^\top \;\big|\;
(u_t z_t)^\top,\ (u_{t-1} z_{t-1})^\top,\ \dots,\ (u_{t-J} z_{t-J})^\top \,\big]^\top
\;\in\; \mathbb{R}^{M}
$$

$$
M = 2(J+1)d_z, \qquad M_G := (J+1)d_z \quad \text{(block split)}
$$

and the stacked input matrix

$$
B \;=\; \big[\, \underbrace{G_0\ G_1\ \cdots\ G_J}_{M_G \text{ columns}} \;\big|\;
\underbrace{\delta G_0\ \delta G_1\ \cdots\ \delta G_J}_{M - M_G \text{ columns}} \,\big]
\;\in\; \mathbb{R}^{p \times M}
$$

Then, identically and with no approximation:

$$
\boxed{\;\sum_{\ell=0}^{J} G_\ell z_{t-\ell} \;+\; \sum_{\ell=0}^{J} u_{t-\ell}\, \delta G_\ell z_{t-\ell}
\;\equiv\; B\, \tilde{u}_t \;}
$$

so the state equation collapses to $s_{t+1} = F s_t + B \tilde{u}_t + m_s + \varepsilon_t$.

**This single identity is why no model class needed rewriting** — the gating lives
entirely in the design matrix, and `ssm`'s built-in input matrix *is* $[G \mid \delta G]$.

---

## 4. Implemented form

The first $J$ bins of each trial are dropped so all lags exist without zero-padding.
With $\tau := t - J$ for $\tau = 0,\dots,T_n - J - 1$, and `ssm`'s AR(1) convention that
places the input at the **destination** index:

$$
\begin{aligned}
s_\tau &\;=\; F s_{\tau-1} + B \tilde{u}_\tau + m_s + \varepsilon_\tau,
&& \varepsilon_\tau \sim \mathcal{N}(0, Q) \\[2pt]
x_\tau &\;\sim\; \operatorname{Poisson}\!\big( \Delta \cdot \exp(C s_\tau + b) \big) \\[2pt]
s_0 &\;\sim\; \mathcal{N}(\mu_{\text{init}}, \Sigma_{\text{init}})
\end{aligned}
$$

This is a one-bin relabeling of §2; $F, B, m_s, Q, C, b$ are unchanged.

**Joint density** (per trial, $T \equiv T_n - J - 1$):

$$
p\big(x_{0:T},\, s_{0:T} \mid \tilde{u}_{0:T}\big)
= p(s_0)
\prod_{\tau=1}^{T} \mathcal{N}\big( s_\tau \,;\, F s_{\tau-1} + B\tilde{u}_\tau + m_s,\ Q \big)
\prod_{\tau=0}^{T} \prod_{i=1}^{d_x} \operatorname{Poisson}\big( x_\tau^{(i)} \,;\, \Delta e^{c_i^\top s_\tau + b_i} \big)
$$

**Frozen emission input.** `ssm`'s generic emission mean is
$\Delta \exp(C s_\tau + F_e \tilde{u}_\tau + b)$. The constraint

$$
F_e \equiv 0
$$

(enforced by removing $F_e$ from the parameter set) is what forces kinematics to act
**only through the latent state**, rather than via a direct kinematics $\to$ spikes path
that would bypass and compete with $s$.

---

## 5. Regression form of the dynamics

Stack regressors and parameters:

$$
r_\tau = \begin{bmatrix} s_{\tau-1} \\ \tilde{u}_\tau \\ 1 \end{bmatrix} \in \mathbb{R}^{D_{\text{in}}},
\qquad
W = \big[\, F \;\big|\; B \;\big|\; m_s \,\big] \in \mathbb{R}^{p \times D_{\text{in}}},
\qquad
D_{\text{in}} = p + M + 1
$$

so that the state equation is simply

$$
s_\tau = W r_\tau + \varepsilon_\tau .
$$

**Column blocks of $W$:**

$$
\underbrace{0 : p-1}_{F} \quad
\underbrace{p : p + M_G - 1}_{G_0 \,\dots\, G_J} \quad
\underbrace{p + M_G : p + M - 1}_{\delta G_0 \,\dots\, \delta G_J} \quad
\underbrace{p + M}_{m_s}
$$

and the per-lag unpacking after fitting:

$$
G_\ell = B[:,\ \ell d_z : (\ell+1) d_z],
\qquad
\delta G_\ell = B[:,\ M_G + \ell d_z : M_G + (\ell+1) d_z]
$$

**Mapping to `ssm` attributes**

| Model | `ssm` attribute | Shape |
|---|---|---|
| $F$ | `dynamics.As[0]` | $p \times p$ |
| $B = [G \mid \delta G]$ | `dynamics.Vs[0]` | $p \times M$ |
| $m_s$ | `dynamics.bs[0]` | $p$ |
| $Q$ | `dynamics.Sigmas[0]` | $p \times p$ |
| $C$ | `emissions.Cs[0]` | $d_x \times p$ |
| $b$ | `emissions.ds[0]` | $d_x$ |

---

## 6. Inference — Laplace-EM

The Poisson likelihood is non-conjugate, so the posterior is approximated by a Gaussian
with block-tridiagonal precision, $q(s) = \mathcal{N}(s\,;\, \hat{s},\, \Lambda_q^{-1})$,
maximizing the evidence lower bound

$$
\mathcal{L}(q, \theta) \;=\; \mathbb{E}_q\!\left[ \log p(x, s \mid \tilde{u}, \theta) \right] + \mathbb{H}[q]
\;\leq\; \log p(x \mid \tilde{u}, \theta)
$$

**E-step.** $\hat{s} = \arg\max_s \log p(x, s \mid \tilde{u}, \theta)$ by Newton's method, with

$$
\Lambda_q \;=\; -\nabla_s^2 \log p(x, s \mid \tilde{u})
\;=\; \underbrace{\vphantom{\sum_\tau} \text{(block-tridiagonal dynamics term)}}_{\text{from } F,\, Q}
\;+\; \sum_\tau C^\top \operatorname{diag}(\lambda_\tau)\, C,
\qquad
\lambda_\tau = \Delta e^{C \hat{s}_\tau + b}
$$

which reduces to exact Kalman smoothing in the linear-Gaussian limit.

**M-step.** Maximize $\mathbb{E}_q[\log p]$ over $\theta$. None of this is re-implemented —
all inference is inherited from `ssm`.

---

## 7. Dynamics M-step — where the per-block ridge enters

Accumulate expected sufficient statistics over all trials and transitions:

$$
S_{rr} = \sum_\tau \mathbb{E}_q\!\left[ r_\tau r_\tau^\top \right] \in \mathbb{R}^{D_{\text{in}} \times D_{\text{in}}},
\qquad
S_{rs} = \sum_\tau \mathbb{E}_q\!\left[ r_\tau s_\tau^\top \right] \in \mathbb{R}^{D_{\text{in}} \times p},
\qquad
S_{ss} = \sum_\tau \mathbb{E}_q\!\left[ s_\tau s_\tau^\top \right] \in \mathbb{R}^{p \times p}
$$

MAP solution under a Gaussian prior $(J_0, h_0)$ on $W$:

$$
\hat{W} \;=\; \big( S_{rs} + h_0 \big)^\top \big( S_{rr} + J_0 \big)^{-1}
$$

The customization sets $h_0 = 0$ (pure shrinkage toward zero, **not** toward the identity
as in `ssm`'s default) and makes the prior precision block-diagonal:

$$
J_0 \;=\; \Lambda \;=\; \operatorname{diag}\Big(
\underbrace{\lambda_F I_p}_{F},\;
\underbrace{\lambda_G I_{M_G}}_{G\text{ block}},\;
\underbrace{\lambda_{\delta G} I_{M - M_G}}_{\delta G\text{ block}},\;
\underbrace{0}_{m_s\ \text{unpenalized}}
\Big)
$$

Equivalently, the M-step maximizes the penalized expected log-likelihood

$$
\sum_\tau \mathbb{E}_q \log \mathcal{N}\big( s_\tau ;\, W r_\tau,\, Q \big)
\;-\; \tfrac{1}{2}\Big[
\lambda_F \|F\|_F^2
+ \lambda_G \sum_{\ell=0}^{J} \|G_\ell\|_F^2
+ \lambda_{\delta G} \sum_{\ell=0}^{J} \|\delta G_\ell\|_F^2
\Big]
$$

> **Why $\lambda_G$ and $\lambda_{\delta G}$ must be separable.** The gated regressors
> $u_{t-\ell} z_{t-\ell}$ are nonzero only on the fraction of bins where $u_t = 1$, so
> $\delta G$ is estimated from far less effective data than $G$ and needs its own
> shrinkage level.

> **Implementation caveat.** For a plain AR(1) dynamics object, `ssm` takes an *exact*
> M-step path that discards $J_0$ and $h_0$ entirely — the penalty would silently do
> nothing. The subclass exists to force the prior-respecting path, and is therefore used
> unconditionally, even when $\lambda_G = \lambda_{\delta G}$.

**Innovation covariance**, MAP under an inverse-Wishart prior $\mathcal{IW}(\nu_0, \Psi_0)$:

$$
E_{\text{sq}} = S_{ss} - \hat{W} S_{rs} - \big(\hat{W} S_{rs}\big)^\top + \hat{W} S_{rr} \hat{W}^\top,
\qquad
\hat{Q} = \frac{E_{\text{sq}} + \Psi_0}{\nu_0 + N_\tau + p + 1}
$$

where $N_\tau$ is the total number of transitions across trials.

---

## 8. Emissions M-step

No closed form. $(C, b)$ maximize the expected Poisson log-likelihood

$$
\sum_\tau \sum_{i=1}^{d_x}
\mathbb{E}_q\!\left[\, x_\tau^{(i)} \big( c_i^\top s_\tau + b_i \big) - \Delta e^{c_i^\top s_\tau + b_i} \,\right]
\;+\; \text{const}
$$

by gradient ascent (autograd), with $F_e$ excluded from the parameter set so it stays $0$.

---

## 9. Nested models as linear restrictions

The three fitted models are the same equations under nested constraints on $B$:

| Model | $M$ | Restriction |
|---|---|---|
| **full** | $2(J+1)d_z$ | $B = [G \mid \delta G]$ unrestricted |
| **no-$\delta G$** | $(J+1)d_z$ | $\delta G_\ell \equiv 0\ \ \forall \ell$ — the null hypothesis |
| **intrinsic** | $0$ | $G_\ell \equiv \delta G_\ell \equiv 0$, i.e. $s_\tau = F s_{\tau-1} + m_s + \varepsilon_\tau$ |

All three are trained and scored on the **same** trimmed observations $x_{J:T}$, so
held-out comparison is like-for-like.

---

## 10. Evaluation

One-step-ahead prediction from the smoothed state, then the Poisson rate:

$$
\hat{s}_\tau^{\text{pred}} = F\, \mathbb{E}_q[s_{\tau-1}] + B \tilde{u}_\tau + m_s,
\qquad
\mu_\tau = \Delta \cdot \exp\!\big( C \hat{s}_\tau^{\text{pred}} + b \big) \in \mathbb{R}^{d_x}
$$

**Metrics** (deviance identical to the existing GLM evaluator):

$$
D(x, \mu) \;=\; 2 \sum_{\tau, i}
\left[\, x_\tau^{(i)} \log \frac{x_\tau^{(i)}}{\mu_\tau^{(i)}} - \big( x_\tau^{(i)} - \mu_\tau^{(i)} \big) \right],
\qquad 0 \log 0 := 0
$$

$$
\ell(x, \mu) \;=\; \sum_{\tau, i} \left[\, x_\tau^{(i)} \log \mu_\tau^{(i)} - \mu_\tau^{(i)} - \log x_\tau^{(i)}! \,\right]
$$

$$
\text{pseudo-}R^2 \;=\; 1 - \frac{D(x, \mu)}{D(x, \bar{\mu})},
\qquad \bar{\mu}^{(i)} = \text{mean count of neuron } i \text{ over held-out bins}
$$

$$
\text{held-out MLL} \;=\; \mathcal{L}(q, \hat{\theta}) \quad \text{(ELBO at fixed } \hat{\theta}\text{)}
$$

**The scientific test.** With folds split by *trial* (never by within-trial bin):

$$
\mathbb{E}_{\text{folds}}\big[ D_{\text{full}} \big] \;<\; \mathbb{E}_{\text{folds}}\big[ D_{\text{no-}\delta G} \big]
$$

is the evidence that the perturbation genuinely modulates the kinematic drive.

---

## 11. Identifiability — which quantities are interpretable

For any invertible $T \in \mathbb{R}^{p \times p}$, the change of variables $s \mapsto T s$
leaves the likelihood **exactly** unchanged under

$$
F \mapsto T F T^{-1}, \quad
B \mapsto T B, \quad
m_s \mapsto T m_s, \quad
Q \mapsto T Q T^\top, \quad
C \mapsto C T^{-1}, \quad
b \mapsto b
$$

So $G_\ell$, $\delta G_\ell$, and $C$ individually are defined only up to this gauge.
Gauge-**invariant** quantities — the ones to report — are:

- $\operatorname{eig}(F)$, the latent timescales;
- the row space of $\delta G_\ell$;
- $C\, \delta G_\ell$, the perturbation effect expressed in **neural** space;
- the null hypothesis $\delta G_\ell = 0$;
- all held-out predictive metrics.

This is why the smoke test checks eigenvalues of $F$ and an affine-regression $R^2$ on the
latents, rather than entrywise parameter recovery.

---

## Code map

| File | Contents |
|---|---|
| `inputs.py` | $\tilde{u}_t$ construction (§3), lag/trim convention (§4) |
| `model.py` | $J_0 = \Lambda$ block ridge (§7), $F_e \equiv 0$ emissions (§4) |
| `fit.py` | LDS assembly, Laplace-EM call (§6), unpacking $B \to G_\ell, \delta G_\ell$ (§5) |
| `evaluate.py` | one-step-ahead rates and deviance (§10) |
| `crossval.py` | trial-level folds, model comparison (§9, §10) |
| `synthetic.py` | recovery checks under the gauge (§11) |
