#!/usr/bin/env python
"""Validate and visualize a fitted perturbation PLDS.

Consumes the ``results.pkl`` written by ``run_smoke_test.py`` and produces:

  * plot_F            -- the latent autonomous matrix F: heatmap + eigenvalue
                         spectrum in the complex plane against the unit circle
                         (overlaid with true F's eigenvalues when synthetic).
  * plot_elbo         -- ELBO / log-marginal-likelihood convergence.
  * plot_input_gains  -- kinematic gain G_l and perturbation gain dG_l per lag,
                         plus a per-lag Frobenius-norm bar chart (dG is the
                         primary scientific object of interest).
  * plot_latents      -- example smoothed latent trajectories (overlaid with the
                         affine-aligned true latents when synthetic).
  * plot_cv           -- held-out deviance per model (the model-selection test).

and prints a textual validation report (stability of F, ELBO monotonicity,
fit quality, and -- for synthetic data -- eigenvalue error & latent R^2).

Usage (use the env that has ssm + matplotlib):
    conda run -n ssm python -m perturbation_plds.validate_results --in smoke_out/results.pkl
    conda run -n ssm python -m perturbation_plds.validate_results --in smoke_out/results.pkl --show
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")          # headless-safe; --show forces a backend later
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_F(F, F_true=None, title="Latent autonomous matrix F"):
    """Heatmap of F next to its eigenvalue spectrum (with the unit circle).

    A discrete-time linear system s_{t+1}=F s_t is stable iff all eigenvalues lie
    strictly inside the unit circle, so the spectrum panel doubles as a stability
    diagnostic.
    """
    ncol = 2 if F_true is None else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.2))

    vmax = np.max(np.abs(F))
    im = axes[0].imshow(F, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("F (fitted)")
    axes[0].set_xlabel("from latent dim")
    axes[0].set_ylabel("to latent dim")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    ax_eig = axes[-1]
    th = np.linspace(0, 2 * np.pi, 256)
    ax_eig.plot(np.cos(th), np.sin(th), "k--", lw=1, label="unit circle")
    ev = np.linalg.eigvals(F)
    ax_eig.scatter(ev.real, ev.imag, c="C0", s=60, zorder=3, label="fitted")
    if F_true is not None:
        evt = np.linalg.eigvals(F_true)
        ax_eig.scatter(evt.real, evt.imag, marker="x", c="C3", s=70, zorder=4,
                       label="true")
        imt = axes[1].imshow(F_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("F (true)")
        axes[1].set_xlabel("from latent dim")
        fig.colorbar(imt, ax=axes[1], fraction=0.046)
    rad = float(np.max(np.abs(ev)))
    ax_eig.set_title("eig(F)  |spectral radius=%.3f|" % rad)
    ax_eig.set_xlabel("Re"); ax_eig.set_ylabel("Im")
    ax_eig.axhline(0, color="0.7", lw=0.6); ax_eig.axvline(0, color="0.7", lw=0.6)
    ax_eig.set_aspect("equal", "box")
    ax_eig.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_elbo(trace, title="ELBO convergence"):
    trace = np.asarray(trace, float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trace, "-o", ms=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log marginal likelihood (ELBO)")
    diffs = np.diff(trace)
    worst = float(diffs.min()) if diffs.size else 0.0
    ax.set_title("%s  (worst step=%.3g)" % (title, worst))
    fig.tight_layout()
    return fig


def plot_input_gains(G_lag, dG_lag, dt=None):
    """Per-lag heatmaps of G_l and dG_l plus a Frobenius-norm-vs-lag bar chart."""
    G_lag = np.asarray(G_lag); dG_lag = np.asarray(dG_lag)
    Jp1 = G_lag.shape[0]
    fig = plt.figure(figsize=(3.0 * Jp1, 9))
    gs = fig.add_gridspec(3, Jp1)
    gmax = max(np.max(np.abs(G_lag)), 1e-12)
    dmax = max(np.max(np.abs(dG_lag)), 1e-12)
    for l in range(Jp1):
        a0 = fig.add_subplot(gs[0, l])
        im0 = a0.imshow(G_lag[l], cmap="RdBu_r", vmin=-gmax, vmax=gmax, aspect="auto")
        a0.set_title("G  lag %d" % l)
        a0.set_xlabel("kinematic dim"); a0.set_ylabel("latent dim")
        fig.colorbar(im0, ax=a0, fraction=0.046)

        a1 = fig.add_subplot(gs[1, l])
        im1 = a1.imshow(dG_lag[l], cmap="PuOr_r", vmin=-dmax, vmax=dmax, aspect="auto")
        a1.set_title(r"$\delta$G  lag %d" % l)
        a1.set_xlabel("kinematic dim"); a1.set_ylabel("latent dim")
        fig.colorbar(im1, ax=a1, fraction=0.046)

    # bottom row: a single norm-vs-lag bar chart spanning all columns
    ax_bar = fig.add_subplot(gs[2, :])
    lags = np.arange(Jp1)
    gn = np.linalg.norm(G_lag.reshape(Jp1, -1), axis=1)
    dn = np.linalg.norm(dG_lag.reshape(Jp1, -1), axis=1)
    w = 0.38
    ax_bar.bar(lags - w / 2, gn, w, label="||G_l||_F", color="C0")
    ax_bar.bar(lags + w / 2, dn, w, label=r"||$\delta$G_l||_F", color="C1")
    ax_bar.set_xlabel("lag l"); ax_bar.set_ylabel("Frobenius norm")
    ax_bar.set_xticks(lags)
    ax_bar.set_title("gain magnitude per lag")
    ax_bar.legend()
    fig.tight_layout()
    return fig


def plot_latents(Es_list, true_s_list=None, J=0, n_show=3, max_dims=4):
    """Smoothed latent trajectories for a few trials.

    When ``true_s_list`` is provided (synthetic data), the true latents are first
    affinely aligned to the smoothed ones (the latent state is identified only up
    to an invertible affine map) and overlaid as dashed lines.
    """
    n_show = min(n_show, len(Es_list))
    p = Es_list[0].shape[1]
    dims = min(max_dims, p)

    W = None
    if true_s_list is not None:
        W = _affine_align(Es_list, true_s_list, J)  # maps Es -> true

    fig, axes = plt.subplots(n_show, 1, figsize=(9, 2.4 * n_show), squeeze=False)
    for i in range(n_show):
        ax = axes[i][0]
        Es = np.asarray(Es_list[i])
        for d in range(dims):
            ax.plot(Es[:, d], color="C%d" % d, lw=1.4, label="latent %d" % d)
        if W is not None:
            s_true = np.asarray(true_s_list[i])[J:]
            T = min(Es.shape[0], s_true.shape[0])
            Es_aug = np.column_stack([Es[:T], np.ones(T)])
            # align fitted into true space so dashed/solid share an axis
            Es_in_true = Es_aug @ W
            for d in range(dims):
                ax.plot(s_true[:T, d], "--", color="C%d" % d, lw=1.0, alpha=0.8)
        ax.set_ylabel("trial %d" % i)
        if i == 0:
            ax.set_title("smoothed latents (solid)" +
                         ("  vs aligned true (dashed)" if W is not None else ""))
        if i == n_show - 1:
            ax.set_xlabel("time bin")
    axes[0][0].legend(loc="upper right", ncol=dims, fontsize=8)
    fig.tight_layout()
    return fig


def _affine_align(Es_list, true_s_list, J):
    """Least-squares affine map [Es, 1] -> true_s (for overlay only)."""
    X, Y = [], []
    for Es, s_true in zip(Es_list, true_s_list):
        s_true = np.asarray(s_true)[J:]
        Es = np.asarray(Es)
        T = min(Es.shape[0], s_true.shape[0])
        X.append(np.column_stack([Es[:T], np.ones(T)]))
        Y.append(s_true[:T])
    X = np.concatenate(X); Y = np.concatenate(Y)
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def plot_cv(cv, metric="deviance_x_per_obs"):
    """Bar chart of held-out CV metric per model (lower deviance is better)."""
    models = list(cv.keys())
    means = [cv[m]["mean"][metric] for m in models]
    stds = [cv[m]["std"].get(metric, 0.0) for m in models]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(models, means, yerr=stds, capsize=4, color="C2")
    ax.set_ylabel("held-out %s" % metric)
    ax.set_title("cross-validated model selection (lower = better)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Numeric validation report
# --------------------------------------------------------------------------- #
def validate(artifact):
    """Return a dict of pass/fail-style diagnostics and print a report."""
    res = artifact["results"]
    F = np.asarray(res["F"])
    ev = np.linalg.eigvals(F)
    rad = float(np.max(np.abs(ev)))
    checks = artifact.get("checks", {})
    metrics = artifact.get("metrics", {})

    print("=" * 64)
    print("VALIDATION REPORT")
    print("=" * 64)
    print("latent dim p=%d   lag J=%d   M(input)=%d"
          % (res["p"], res["J"], res["_meta"]["M"]))
    print("-" * 64)
    print("F stability      : spectral radius=%.3f  -> %s"
          % (rad, "STABLE" if rad < 1.0 else "UNSTABLE (>=1)"))
    if "elbo_monotone" in checks:
        print("ELBO monotone    : %s" % checks["elbo_monotone"])
    if metrics:
        print("in-sample fit    : deviance/obs=%.5f  pseudo-R2=%.4f"
              % (metrics.get("deviance_x_per_obs", float("nan")),
                 metrics.get("pseudoR2_x", float("nan"))))
    if artifact.get("is_synthetic"):
        print("-- synthetic ground-truth checks --")
        print("eig(F) error     : %s  (smaller is better)"
              % _fmt(checks.get("eig_error")))
        r2 = checks.get("latent_r2")
        print("latent affine R2 : %s  -> %s"
              % (_fmt(r2), "PASS (>0.7)" if (r2 is not None and r2 > 0.7) else "review"))
    cv = artifact.get("cv")
    if cv and "full" in cv and "no_dG" in cv:
        df = cv["full"]["mean"]["deviance_x_per_obs"]
        dn = cv["no_dG"]["mean"]["deviance_x_per_obs"]
        print("-- model selection (held-out) --")
        print("full=%.5f  no_dG=%.5f  -> dG helps: %s" % (df, dn, df < dn))
    print("=" * 64)
    return dict(spectral_radius=rad, eig=ev)


def _fmt(v):
    return "n/a" if v is None else ("%.4f" % v)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join("smoke_out", "results.pkl"),
                    help="path to results.pkl from run_smoke_test.py")
    ap.add_argument("--out", default=None,
                    help="dir for PNGs (default: <results dir>/figures)")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    ap.add_argument("--n-latent-trials", type=int, default=3)
    args = ap.parse_args(argv)

    if args.show:
        matplotlib.use(matplotlib.get_backend())  # keep an interactive backend
    art = load_artifact(args.inp)
    res = art["results"]
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.inp)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    validate(art)

    true = art.get("true")
    figs = {}
    figs["F"] = plot_F(res["F"], F_true=(true["F"] if true else None))
    figs["elbo"] = plot_elbo(res["log_marginal_likelihood_trace"])
    figs["input_gains"] = plot_input_gains(res["G_lag"], res["dG_lag"], dt=res["dt"])
    figs["latents"] = plot_latents(
        art["Es_list"], true_s_list=(true["s_list"] if true else None),
        J=res["J"], n_show=args.n_latent_trials)
    if art.get("cv"):
        figs["cv"] = plot_cv(art["cv"])

    for name, fig in figs.items():
        path = os.path.join(out_dir, "%s.png" % name)
        fig.savefig(path, dpi=130)
        print("saved %s" % path)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
