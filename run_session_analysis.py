"""End-to-end analysis of one recording session with the customized
perturbation PLDS (``perturbation_plds`` package).

Pipeline
--------
1. Load a session ``.pkl`` and build the model inputs ``Z`` (kinematics),
   ``X`` (spike counts), ``U`` (binary perturbation), exactly as in the user's
   notebook.
2. Summarize the data (shapes, perturbation timing, firing statistics).
3. Fit the full perturbation PLDS and write a markdown report with diagnostic
   figures (ELBO, eig(F), per-lag ||G_l|| and ||dG_l||, latent trajectories).
4. Run trial-level cross-validation comparing the full model against the
   ``no_dG`` and ``intrinsic`` ablations to test whether G (kinematics) and
   dG (perturbation-gated kinematics) actually improve held-out prediction.
   The report is updated with the CV results.

The report and figures are written incrementally so partial progress survives
an interruption.  Run from the repo root:

    python run_session_analysis.py            # session 0, default config
    python run_session_analysis.py --session 1 --p 8 --J 3

Heavy step (Laplace-EM) costs ~14 s/iter on the full 140-trial session, so the
default iteration budgets are tuned for a ~1 hour total run.  The expensive
cross-validation can be skipped with ``--no-cv``.
"""

import argparse
import os
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perturbation_plds import (
    fit_perturbation_plds,
    evaluate_perturbation_plds,
    crossval_compare_models,
)

DT = 5.0 / 240.0
DATA_TMPL = ("/home/sp645/isilon/All_Staff/sp645/data/Yi_deconvoled_calcium/"
             "Ephys_no_smooth/session{n}.pkl")


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_session(session_num, selected_window=168):
    """Load a session and build Z (kinematics), X (counts), U (perturbation).

    Mirrors the user's notebook: Z = [pos_x, pos_y, vel_x, vel_y] (velocity is
    the first difference of position, last bin zero-padded), U = behavior
    channel 2 (binary perturbation), X = the spike-count train array.
    """
    with open(DATA_TMPL.format(n=session_num), "rb") as f:
        obj = pickle.load(f)
    beh = obj["behavior"][:, :selected_window, :]
    train = obj["train"][:, :selected_window, :]

    Y = beh[:, :, :2]
    velocity = Y[:, 1:, :] - Y[:, :-1, :]
    velocity = np.concatenate(
        [velocity, np.zeros((train.shape[0], 1, 2))], axis=1)
    Z = np.concatenate([Y, velocity], axis=-1)   # (N, T, 4)
    U = beh[:, :, 2]                              # (N, T)
    X = train                                     # (N, T, d_x)
    return Z, U, X


# --------------------------------------------------------------------------- #
# Markdown report helper                                                       #
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self, path):
        self.path = path
        self._buf = []

    def add(self, text=""):
        self._buf.append(text)
        self.flush()

    def flush(self):
        with open(self.path, "w") as f:
            f.write("\n".join(self._buf) + "\n")


# --------------------------------------------------------------------------- #
# Data analysis                                                               #
# --------------------------------------------------------------------------- #
def analyze_data(Z, U, X, rep, outdir):
    N, T, d_z = Z.shape
    d_x = X.shape[2]
    kin_names = ["pos_x", "pos_y", "vel_x", "vel_y"][:d_z]

    perturbed = (U > 0).any(axis=1)
    onsets = np.array([np.argmax(U[i] > 0) for i in range(N) if U[i].any()])
    mean_count = X.mean()
    per_neuron_rate = X.mean(axis=(0, 1))

    rep.add("## 1. Data summary\n")
    rep.add(f"- **Trials (N):** {N}  |  **Bins/trial (T):** {T}  "
            f"|  **bin width Δ:** {DT*1000:.2f} ms")
    rep.add(f"- **Kinematic dim d_z:** {d_z}  ({', '.join(kin_names)})  "
            f"— *note: real data has d_z={d_z}, not the spec's 6; "
            "the model infers d_z automatically.*")
    rep.add(f"- **Neuron / channel dim d_x:** {d_x}")
    rep.add("")
    rep.add("**Kinematics (Z)** — mean ± std per dim:")
    rep.add("")
    rep.add("| dim | mean | std |")
    rep.add("|-----|------|-----|")
    for j, nm in enumerate(kin_names):
        rep.add(f"| {nm} | {Z[:,:,j].mean():.3f} | {Z[:,:,j].std():.3f} |")
    rep.add("")
    rep.add("**Perturbation (U)** — binary {0,1}:")
    rep.add(f"- fraction of all bins perturbed: **{(U>0).mean():.3f}**")
    rep.add(f"- trials with any perturbation: **{int(perturbed.sum())}/{N}**")
    if len(onsets):
        rep.add(f"- onset bin (within trial): min {onsets.min()}, "
                f"median {int(np.median(onsets))}, max {onsets.max()}")
        rep.add(f"- mean perturbed bins per perturbed trial: "
                f"**{(U>0).sum(1)[perturbed].mean():.1f}**")
    rep.add("")
    rep.add("**Spike counts (X)** — Poisson observations:")
    rep.add(f"- integer counts: {np.allclose(X, np.round(X))}, "
            f"range [{int(X.min())}, {int(X.max())}]")
    rep.add(f"- mean count/bin: **{mean_count:.3f}**, "
            f"fraction of zeros: **{(X==0).mean():.3f}**")
    rep.add(f"- per-neuron mean count: min {per_neuron_rate.min():.4f}, "
            f"max {per_neuron_rate.max():.4f}, "
            f"neurons with zero total count: {int((X.sum((0,1))==0).sum())}")
    rep.add("")

    # figure: data overview
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(per_neuron_rate, bins=30, color="steelblue")
    ax[0].set(title="Per-neuron mean count/bin", xlabel="mean count",
              ylabel="# neurons")
    ax[1].plot(U.mean(axis=0), color="firebrick")
    ax[1].set(title="Perturbation prevalence over time",
              xlabel="bin", ylabel="frac trials perturbed")
    ax[2].plot(X.mean(axis=(0, 2)), color="seagreen")
    ax[2].set(title="Population mean count over time",
              xlabel="bin", ylabel="mean count")
    fig.tight_layout()
    fpath = os.path.join(outdir, "data_overview.png")
    fig.savefig(fpath, dpi=110)
    plt.close(fig)
    rep.add(f"![data overview]({os.path.basename(fpath)})\n")


# --------------------------------------------------------------------------- #
# Fit diagnostics / report                                                    #
# --------------------------------------------------------------------------- #
def report_fit(res, Z, U, X, rep, outdir, cfg, fit_seconds):
    p, J, d_z = res["p"], res["J"], res["_meta"]["d_z"]
    F = res["F"]
    G = res["G_lag"]    # (J+1, p, d_z)
    dG = res["dG_lag"]  # (J+1, p, d_z)
    elbo = res["log_marginal_likelihood_trace"]
    eig = np.linalg.eigvals(F)
    eig_mag = np.sort(np.abs(eig))[::-1]

    G_norm = np.linalg.norm(G, axis=(1, 2))    # per lag
    dG_norm = np.linalg.norm(dG, axis=(1, 2))
    ratio = dG_norm / np.maximum(G_norm, 1e-12)

    rep.add("## 2. Model fit\n")
    rep.add(f"- **Latent dim p:** {p}  |  **lag J:** {J}  "
            f"(input dim M = 2·(J+1)·d_z = {2*(J+1)*d_z})")
    rep.add(f"- **Ridge:** lam_G = {cfg['lam_G']:g}, "
            f"lam_dG = {cfg['lam_dG']:g}, lam_F = {cfg['lam_F']:g}")
    rep.add(f"- **Laplace-EM:** {cfg['n_iters']} iters "
            f"(+{cfg['num_init_iters']} init), wall time {fit_seconds/60:.1f} min")
    rep.add(f"- **Final ELBO:** {elbo[-1]:,.1f}  "
            f"(start {elbo[0]:,.1f}, total ascent {elbo[-1]-elbo[0]:,.1f})")
    drops = np.diff(elbo)
    rep.add(f"- **ELBO monotonicity:** max single-step drop "
            f"{(-drops.min() if (drops<0).any() else 0.0):.2f} "
            f"({'non-decreasing' if (drops>=-1e-3*abs(elbo[-1])).all() else 'has dips'})")
    rep.add("")
    rep.add("**Dynamics eigenvalues |λ(F)|** (top 6):")
    rep.add("`" + ", ".join(f"{v:.3f}" for v in eig_mag[:6]) + "`")
    rep.add(f"- spectral radius: **{eig_mag[0]:.3f}** "
            f"({'stable' if eig_mag[0] < 1 else 'UNSTABLE'})")
    rep.add("")

    rep.add("### Kinematic gain G and perturbation modulation δG\n")
    rep.add("Frobenius norm of each lag block "
            "(||·|| over the p×d_z matrix):\n")
    rep.add("| lag ℓ | ‖G_ℓ‖ | ‖δG_ℓ‖ | ‖δG_ℓ‖/‖G_ℓ‖ |")
    rep.add("|------|-------|--------|---------------|")
    for l in range(J + 1):
        rep.add(f"| {l} | {G_norm[l]:.3f} | {dG_norm[l]:.3f} "
                f"| {ratio[l]:.3f} |")
    rep.add("")
    rep.add(f"- total ‖G‖ = {np.linalg.norm(G):.3f}, "
            f"total ‖δG‖ = {np.linalg.norm(dG):.3f}, "
            f"overall ratio ‖δG‖/‖G‖ = "
            f"{np.linalg.norm(dG)/max(np.linalg.norm(G),1e-12):.3f}")
    rep.add("")
    rep.add("> ‖δG_ℓ‖ measures how strongly the perturbation **re-weights** the "
            "way lagged kinematics drive the latent neural state. A non-trivial "
            "ratio is the first (in-sample) sign that the perturbation gates "
            "kinematic input; cross-validation (section 3) tests whether this is "
            "real or overfitting.\n")

    # ---- figures ----
    # ELBO
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbo, marker=".")
    ax.set(title="Laplace-EM ELBO", xlabel="iteration", ylabel="ELBO")
    fig.tight_layout()
    f1 = os.path.join(outdir, "elbo.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)

    # eigenvalues in complex plane
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    ax.scatter(eig.real, eig.imag, c="crimson", zorder=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set(title="Eigenvalues of F", xlabel="Re", ylabel="Im")
    ax.set_aspect("equal")
    fig.tight_layout()
    f2 = os.path.join(outdir, "eig_F.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)

    # per-lag norms
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(J + 1)
    ax.bar(x - 0.2, G_norm, width=0.4, label="‖G_ℓ‖", color="steelblue")
    ax.bar(x + 0.2, dG_norm, width=0.4, label="‖δG_ℓ‖", color="darkorange")
    ax.set(title="Kinematic gain vs perturbation modulation by lag",
           xlabel="lag ℓ", ylabel="Frobenius norm")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    f3 = os.path.join(outdir, "G_dG_norms.png")
    fig.savefig(f3, dpi=110)
    plt.close(fig)

    # G_0 / dG_0 heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    vmax = max(np.abs(G[0]).max(), np.abs(dG[0]).max())
    for a, M, ttl in ((ax[0], G[0], "G_0"), (ax[1], dG[0], "δG_0")):
        im = a.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set(title=ttl, xlabel="kinematic dim", ylabel="latent dim")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    f4 = os.path.join(outdir, "G0_dG0_heatmap.png")
    fig.savefig(f4, dpi=110)
    plt.close(fig)

    # latent trajectories: smoothed states, mark perturbation
    lds = res["_ssm_model"]
    from perturbation_plds.inputs import build_augmented_inputs
    u_aug, x_trim, _z, _m = build_augmented_inputs(Z, U, J, x_list=X)
    _e, post = lds.approximate_posterior(
        x_trim, inputs=u_aug, method="laplace_em", num_iters=15, verbose=0)
    Es = post.mean_continuous_states
    fig, ax = plt.subplots(min(3, p), 1, figsize=(9, 6), sharex=True)
    if p == 1:
        ax = [ax]
    trial_ids = [0, 1, 2]
    for d in range(min(3, p)):
        for tid in trial_ids:
            ax[d].plot(Es[tid][:, d], lw=1, label=f"trial {tid}" if d == 0 else None)
        # shade perturbation for first trial
        u0 = U[trial_ids[0]][J:]
        ax[d].fill_between(np.arange(len(u0)), ax[d].get_ylim()[0],
                           ax[d].get_ylim()[1], where=u0 > 0, color="red",
                           alpha=0.08)
        ax[d].set_ylabel(f"latent {d}")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("bin (trial-aligned, first J dropped)")
    ax[0].set_title("Smoothed latent trajectories "
                    "(red shade = perturbation in trial 0)")
    fig.tight_layout()
    f5 = os.path.join(outdir, "latents.png")
    fig.savefig(f5, dpi=110)
    plt.close(fig)

    for fp in (f1, f2, f3, f4, f5):
        rep.add(f"![{os.path.basename(fp)}]({os.path.basename(fp)})")
    rep.add("")

    # in-sample evaluation
    metrics = evaluate_perturbation_plds(res, Z, X, U)
    rep.add("### In-sample one-step-ahead prediction\n")
    rep.add(f"- Poisson deviance / obs: **{metrics['deviance_x_per_obs']:.4f}**")
    rep.add(f"- pseudo-R² (vs constant-rate null): "
            f"**{metrics['pseudoR2_x']:.4f}**")
    rep.add(f"- log-likelihood: {metrics['loglik_x']:,.1f} "
            f"over {metrics['n_obs']:,} observations")
    rep.add("")
    return metrics


# --------------------------------------------------------------------------- #
# Cross-validation                                                            #
# --------------------------------------------------------------------------- #
def report_cv(Z, U, X, rep, cfg, outdir):
    rep.add("## 3. Cross-validation: are G and δG real?\n")
    rep.add("Trial-level k-fold CV (folds split by **trial**, never by bin). "
            "Three nested models share the same Laplace-EM budget; lower "
            "held-out Poisson deviance/obs is better.\n")
    rep.add("- **full** : latent + lagged kinematics G + perturbation-gated δG")
    rep.add("- **no_dG** : latent + lagged kinematics G only (δG ≡ 0)")
    rep.add("- **intrinsic** : latent dynamics only (no kinematic input)")
    rep.add("")

    t0 = time.time()
    cv = crossval_compare_models(
        Z, X, U, p=cfg["p"], J=cfg["J"], n_folds=cfg["n_folds"],
        lam_G=cfg["lam_G"], lam_dG=cfg["lam_dG"], lam_F=cfg["lam_F"],
        dt=DT, n_iters=cfg["cv_n_iters"], num_init_iters=cfg["cv_num_init_iters"],
        seed=0, verbose=0, models=("intrinsic", "no_dG", "full"))
    cv_seconds = time.time() - t0

    rep.add(f"*({cfg['n_folds']} folds, {cfg['cv_n_iters']} iters/fit, "
            f"wall time {cv_seconds/60:.1f} min)*\n")
    rep.add("| model | held-out deviance/obs (mean ± std) | "
            "pseudo-R² | held-out ELBO |")
    rep.add("|-------|-----------------------------------|-----------|------------|")
    order = ["intrinsic", "no_dG", "full"]
    dev = {}
    for m in order:
        mn = cv[m]["mean"]
        sd = cv[m]["std"]
        dev[m] = mn["deviance_x_per_obs"]
        rep.add(f"| {m} | {mn['deviance_x_per_obs']:.4f} ± "
                f"{sd['deviance_x_per_obs']:.4f} | {mn['pseudoR2_x']:.4f} "
                f"| {mn['held_out_marginal_likelihood']:,.0f} |")
    rep.add("")

    # verdicts
    g_gain = dev["intrinsic"] - dev["no_dG"]
    dg_gain = dev["no_dG"] - dev["full"]
    rep.add("### Verdict\n")
    rep.add(f"- **G (kinematics) helps?** intrinsic − no_dG = "
            f"`{g_gain:+.4f}` deviance/obs → "
            f"**{'YES' if g_gain > 0 else 'no'}** "
            f"(kinematics {'improve' if g_gain>0 else 'do not improve'} "
            "held-out prediction).")
    rep.add(f"- **δG (perturbation gating) helps?** no_dG − full = "
            f"`{dg_gain:+.4f}` deviance/obs → "
            f"**{'YES' if dg_gain > 0 else 'no'}** "
            f"(perturbation-gated kinematics "
            f"{'improve' if dg_gain>0 else 'do not improve'} held-out "
            "prediction beyond plain kinematics).")
    rep.add("")
    rep.add("> The δG comparison is the key scientific test: if **full** beats "
            "**no_dG** out of sample, the proprioceptive perturbation genuinely "
            "changes how kinematics drive the latent neural state (δG ≠ 0), "
            "not merely as an in-sample artifact.\n")

    # CV bar figure
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(order))
    means = [cv[m]["mean"]["deviance_x_per_obs"] for m in order]
    stds = [cv[m]["std"]["deviance_x_per_obs"] for m in order]
    ax.bar(xs, means, yerr=stds, capsize=5,
           color=["gray", "steelblue", "darkorange"])
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set(title="Held-out Poisson deviance / obs (lower = better)",
           ylabel="deviance / obs")
    ax.set_ylim(min(means) - 2 * max(stds), max(means) + 2 * max(stds))
    fig.tight_layout()
    fp = os.path.join(outdir, "cv_deviance.png")
    fig.savefig(fp, dpi=110)
    plt.close(fig)
    rep.add(f"![cv deviance]({os.path.basename(fp)})\n")
    return cv


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--J", type=int, default=3)
    ap.add_argument("--lam_G", type=float, default=1e-3)
    ap.add_argument("--lam_dG", type=float, default=1e-3)
    ap.add_argument("--lam_F", type=float, default=0.0)
    ap.add_argument("--n_iters", type=int, default=60)
    ap.add_argument("--num_init_iters", type=int, default=12)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--cv_n_iters", type=int, default=35)
    ap.add_argument("--cv_num_init_iters", type=int, default=8)
    ap.add_argument("--no-cv", dest="no_cv", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "perturbation_plds_results", f"session{args.session}")
    os.makedirs(outdir, exist_ok=True)
    rep = Report(os.path.join(outdir, "report.md"))

    cfg = vars(args).copy()

    rep.add(f"# Perturbation PLDS — session {args.session}\n")
    rep.add(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by "
            "`run_session_analysis.py`.*\n")

    print("[load] reading session", args.session, flush=True)
    Z, U, X = load_session(args.session, args.window)
    analyze_data(Z, U, X, rep, outdir)

    print("[fit] main model p=%d J=%d ..." % (args.p, args.J), flush=True)
    t0 = time.time()
    res = fit_perturbation_plds(
        Z, X, U, p=args.p, J=args.J, dt=DT,
        lam_G=args.lam_G, lam_dG=args.lam_dG, lam_F=args.lam_F,
        n_iters=args.n_iters, num_init_iters=args.num_init_iters, verbose=2)
    fit_seconds = time.time() - t0
    print("[fit] done in %.1f min" % (fit_seconds / 60), flush=True)

    # persist fitted parameters (drop the unpicklable-heavy ssm model handle)
    save = {k: v for k, v in res.items() if k != "_ssm_model"}
    np.savez(os.path.join(outdir, "fit_params.npz"),
             **{k: v for k, v in save.items()
                if isinstance(v, np.ndarray)})

    report_fit(res, Z, U, X, rep, outdir, cfg, fit_seconds)

    if not args.no_cv:
        print("[cv] cross-validating ...", flush=True)
        report_cv(Z, U, X, rep, cfg, outdir)
        print("[cv] done", flush=True)

    rep.add("---")
    rep.add("\n## Configuration\n")
    rep.add("```")
    for k, v in cfg.items():
        rep.add(f"{k} = {v}")
    rep.add("```")
    rep.flush()
    print("[done] report at", os.path.join(outdir, "report.md"), flush=True)


if __name__ == "__main__":
    main()
