"""Standalone manifold / dimensionality validation for the perturbation PLDS.

Runs the same three analyses as ``validate_fit_results.ipynb`` but as a plain
script (no notebook kernel needed): loads the cached p=10 fit, builds the
unperturbed reference manifold, and prints the stats + writes the three figures.

What each test uses
-------------------
* The **manifold is NOT built from the model parameters**.  It is PCA (top-k PCs,
  90% var) on the *inferred* latents ``E[s_t]`` of a *training half* of the
  unperturbed trials -- the PLDS analog of "calcium after PCA".  Those latents
  were produced once by the smoother in ``fit_p10_cache.py`` and cached.
* **A / B** are pure geometry on the inferred latents (params not used directly).
* **C** is the only test that uses ``F, G, dG, m_s`` -- the counterfactual rollout.

Run:
    /home/sp645/miniconda3/envs/ssm/bin/python run_manifold_analysis.py
(needs the ssm env only for the import of perturbation_plds; the math is numpy.)
"""

import pickle

import matplotlib
matplotlib.use("Agg")          # headless: write figures, never block on a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from perturbation_plds import manifold_analysis as ma

CACHE = "perturbation_plds_results/session0/fit_p10.pkl"
OUTDIR = "perturbation_plds_results/session0"
W_PRE, W_POST = 20, 60         # onset-aligned window (bins before / after onset)
REACH_ONSET = 48               # raw-frame reach onset (1.0 s), same for all trials

# the 8 participation-ratio groups (key, human label, bar colour), in plot order
PR_GROUPS = [
    ("unpert",            "unperturbed",              "tab:blue"),
    ("pert",              "perturbed",                "tab:red"),
    ("unpert_reach_pre",  "unpert reach-pre",         "#9ecae1"),
    ("unpert_reach_post", "unpert reach-post",        "#3182bd"),
    ("pert_reach_pre",    "pert reach-pre",           "#fcae91"),
    ("pert_reach_post",   "pert reach-post",          "#de2d26"),
    ("pert_perturb_pre",  "pert perturb-pre",         "#c7e9c0"),
    ("pert_perturb_post", "pert perturb-post",        "#31a354"),
]


def participation_ratio_8groups(latents, pert, unpert, onsets, J, seed=1):
    """Equal-N bootstrap participation ratio for the 8 condition/epoch groups.

    Two onset references:
      * **reach onset** -- fixed at raw bin ``REACH_ONSET`` (1.0 s), i.e. latent
        index ``REACH_ONSET - J`` (latents are trimmed by ``J``); same split for
        unperturbed and perturbed trials;
      * **perturbation onset** -- per-trial ``onsets[i]`` (already trimmed),
        defined only for perturbed trials.

    All groups are bootstrapped at the same trial count ``n_eq`` so trial count
    is not a confound (the per-epoch *timepoint* count still differs by design).
    Returns ``(groups, reach, n_eq)`` where ``groups`` maps key -> (mean, lo, hi).
    """
    reach = REACH_ONSET - J
    n_eq = min(len(pert), len(unpert))

    def boot(arrs):
        return ma.participation_ratio_bootstrap(arrs, n_trials=n_eq, seed=seed)

    groups = {
        "unpert":            boot([latents[i] for i in unpert]),
        "pert":              boot([latents[i] for i in pert]),
        "unpert_reach_pre":  boot([latents[i][:reach] for i in unpert]),
        "unpert_reach_post": boot([latents[i][reach:] for i in unpert]),
        "pert_reach_pre":    boot([latents[i][:reach] for i in pert]),
        "pert_reach_post":   boot([latents[i][reach:] for i in pert]),
        "pert_perturb_pre":  boot([latents[i][:onsets[i]] for i in pert if onsets[i] > 5]),
        "pert_perturb_post": boot([latents[i][onsets[i]:] for i in pert if onsets[i] > 5]),
    }
    return groups, reach, n_eq


def load_cache():
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    return cache


def build_manifold(latents, unpert, seed=0):
    """Top-k (90% var) PCA subspace of a *training half* of unperturbed latents.

    Splitting unperturbed trials into train/test avoids circularity: held-out
    unperturbed trials should stay on-manifold, perturbed ones should leave.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unpert)
    n_tr = len(perm) // 2
    unp_train, unp_test = list(perm[:n_tr]), list(perm[n_tr:])
    manifold = ma.fit_manifold([latents[i] for i in unp_train], var_frac=0.90)
    return manifold, unp_train, unp_test


def _stack_to_T(arrs):
    T = min(a.shape[0] for a in arrs)
    return np.vstack([a[:T] for a in arrs])


def test_A(latents, pert, unp_test, onsets, manifold, rel, outdir=OUTDIR):
    """A. Off-manifold angle: perturbed vs (held-out) unperturbed, time-resolved."""
    ang_pert = ma.off_manifold_angle([latents[i] for i in pert], manifold)
    ang_unp = ma.off_manifold_angle([latents[i] for i in unp_test], manifold)
    Ap, Au = _stack_to_T(ang_pert), _stack_to_T(ang_unp)
    t = np.arange(Ap.shape[1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for A, lbl, c in [(Au, "unperturbed (held-out)", "tab:blue"),
                      (Ap, "perturbed", "tab:red")]:
        m = np.degrees(A.mean(0)); se = np.degrees(A.std(0) / np.sqrt(A.shape[0]))
        ax[0].plot(t, m, color=c, label=lbl)
        ax[0].fill_between(t, m - se, m + se, color=c, alpha=.2)
    ax[0].set(xlabel="trial time (bin)", ylabel="off-manifold angle (deg)",
              title="Angle off unperturbed manifold vs trial time"); ax[0].legend()

    M = np.full((len(pert), len(rel)), np.nan)
    for r, i in enumerate(pert):
        o, a = onsets[i], ang_pert[r]
        for j, dd in enumerate(rel):
            if 0 <= o + dd < a.shape[0]:
                M[r, j] = np.degrees(a[o + dd])
    mm = np.nanmean(M, 0); ss = np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
    ax[1].plot(rel, mm, color="tab:red")
    ax[1].fill_between(rel, mm - ss, mm + ss, color="tab:red", alpha=.2)
    ax[1].axvline(0, ls="--", c="k", lw=1, label="perturbation onset")
    ax[1].set(xlabel="bins from onset", ylabel="off-manifold angle (deg)",
              title="Perturbed trials, onset-aligned"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/validate_A_angle.png", dpi=110); plt.close(fig)

    # statistics on per-trial mean angle
    post = np.degrees(np.array([np.nanmean(ang_pert[r][onsets[i]:onsets[i] + W_POST])
                                for r, i in enumerate(pert)]))
    unp_mean = np.degrees(np.array([a.mean() for a in ang_unp]))
    _, pmw = mannwhitneyu(post, unp_mean, alternative="greater")
    pre = np.degrees(np.array([np.nanmean(ang_pert[r][max(0, onsets[i] - W_PRE):onsets[i]])
                               for r, i in enumerate(pert)]))
    _, pw = wilcoxon(post, pre, alternative="greater")
    print("\n--- A. off-manifold angle ---")
    print(f"perturbed post-onset   : {post.mean():.2f} +/- {post.std():.2f} deg")
    print(f"unperturbed (held-out) : {unp_mean.mean():.2f} +/- {unp_mean.std():.2f} deg")
    print(f"  perturbed > unperturbed (Mann-Whitney, 1-sided) p = {pmw:.2e}")
    print(f"perturbed pre-onset    : {pre.mean():.2f} +/- {pre.std():.2f} deg")
    print(f"  post > pre within trial (Wilcoxon, 1-sided)     p = {pw:.2e}")
    return dict(post=post.mean(), post_sd=post.std(), unp=unp_mean.mean(),
                unp_sd=unp_mean.std(), pre=pre.mean(), pre_sd=pre.std(),
                p_pert_vs_unp=pmw, p_post_vs_pre=pw)


def test_B(latents, pert, unpert, onsets, manifold, k, J, outdir=OUTDIR):
    """B. Effective dimensionality (participation ratio), 8 groups, equal-N boot.

    Groups split perturbed/unperturbed trials by **reach onset** (fixed bin) and
    perturbed trials by **perturbation onset** (per trial); see
    :func:`participation_ratio_8groups`.
    """
    groups, reach, n_eq = participation_ratio_8groups(latents, pert, unpert, onsets, J)
    off_var = ma.offmanifold_variance_fraction([latents[i] for i in pert], manifold)

    print("\n--- B. effective dimensionality (participation ratio), 8 groups ---")
    print(f"reach onset latent index = {reach} (raw bin {REACH_ONSET} - J {J}); n_eq = {n_eq}")
    for key, lbl, _c in PR_GROUPS:
        m, lo, hi = groups[key]
        print(f"PR {lbl:<20s}: {m:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"fraction of perturbed variance OUTSIDE the {k}-D manifold: {off_var:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(PR_GROUPS))
    vals = [groups[key] for key, _l, _c in PR_GROUPS]
    ax[0].bar(xs, [v[0] for v in vals],
              yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
              color=[c for _k, _l, c in PR_GROUPS], capsize=3)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([l for _k, l, _c in PR_GROUPS], rotation=40, ha="right")
    ax[0].set(ylabel="participation ratio", title="Effective latent dimension (95% CI)")
    for arrs, lbl, c in [([latents[i] for i in unpert], "unperturbed", "tab:blue"),
                         ([latents[i] for i in pert], "perturbed", "tab:red")]:
        cv = ma.cumulative_variance(arrs)
        ax[1].plot(np.arange(1, len(cv) + 1), cv, "-o", color=c, label=lbl)
    ax[1].axhline(0.9, ls=":", c="k")
    ax[1].set(xlabel="# latent PCs", ylabel="cumulative variance",
              title="Variance spectrum"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/validate_B_dim.png", dpi=110); plt.close(fig)

    out = dict(groups=groups, reach=reach, n_eq=n_eq, off_var=off_var, k=k)
    # backward-compatible aliases used by the existing per-session report
    out.update(pr_unp=groups["unpert"], pr_pert=groups["pert"],
               pr_pre=groups["pert_perturb_pre"], pr_post=groups["pert_perturb_post"])
    return out


def test_C(cache, latents, pert, onsets, manifold, k, rel, outdir=OUTDIR):
    """C. Mechanism: counterfactual dG-on vs dG-off rollout (uses model params)."""
    F, m_s = cache["F"], cache["m_s"]
    G_lag, dG_lag = cache["G_lag"], cache["dG_lag"]
    Z_full, U_full = cache["Z_full"], cache["U_full"]
    P = manifold["basis"] @ manifold["basis"].T

    delta_norm = np.full((len(pert), len(rel)), np.nan)
    delta_off = np.full((len(pert), len(rel)), np.nan)
    all_delta = []
    for r, i in enumerate(pert):
        s0 = latents[i][0]
        _s_on, _s_off, delta = ma.counterfactual_latents(
            F, G_lag, dG_lag, m_s, Z_full[i], U_full[i], s0)
        all_delta.append(delta)
        nrm = np.linalg.norm(delta, axis=1)
        perp = delta - delta @ P.T
        frac = np.sum(perp ** 2, 1) / np.clip(np.sum(delta ** 2, 1), 1e-9, None)
        o = onsets[i]
        for j, dd in enumerate(rel):
            if 0 <= o + dd < delta.shape[0]:
                delta_norm[r, j] = nrm[o + dd]; delta_off[r, j] = frac[o + dd]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    mm = np.nanmean(delta_norm, 0); ss = np.nanstd(delta_norm, 0) / np.sqrt(np.sum(~np.isnan(delta_norm), 0))
    ax[0].plot(rel, mm, color="tab:purple")
    ax[0].fill_between(rel, mm - ss, mm + ss, color="tab:purple", alpha=.2)
    ax[0].axvline(0, ls="--", c="k", lw=1)
    ax[0].set(xlabel="bins from onset", ylabel=r"$\|\Delta s\|$",
              title="Causal perturbation effect on latent (counterfactual)")
    mo = np.nanmean(delta_off, 0) * 100
    ax[1].plot(rel, mo, color="tab:green"); ax[1].axvline(0, ls="--", c="k", lw=1)
    ax[1].set(xlabel="bins from onset", ylabel=r"% of $\Delta s$ off unpert. manifold",
              title=r"$\Delta s$ lives off the unperturbed manifold")
    plt.tight_layout(); plt.savefig(f"{outdir}/validate_C_mechanism.png", dpi=110); plt.close(fig)

    post_mask = rel >= 0
    dn_post = np.nanmean(delta_norm[:, post_mask]); dn_pre = np.nanmean(delta_norm[:, ~post_mask])
    drives = [ma.realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="dG") for i in pert]
    dr = np.vstack(drives); dr = dr[np.linalg.norm(dr, axis=1) > 1e-9]
    dr_perp = dr - dr @ P.T
    off_post = np.nanmean(delta_off[:, post_mask]) * 100
    dg_off = 100 * np.sum(dr_perp ** 2) / np.sum(dr ** 2)
    pr_delta = ma.participation_ratio(all_delta)
    print("\n--- C. mechanism (counterfactual) ---")
    print(f"||delta s||  pre-onset {dn_pre:.3f} (0 by construction: u=0 -> no dG drive)"
          f"  ->  post-onset {dn_post:.3f}")
    print(f"fraction of counterfactual effect off manifold (post): {off_post:.1f}%")
    print(f"realized dG drive off the {k}-D manifold: {dg_off:.1f}%")
    print(f"participation ratio of the counterfactual effect delta s: {pr_delta:.2f}")
    return dict(dn_pre=dn_pre, dn_post=dn_post, off_post=off_post,
                dg_off=dg_off, pr_delta=pr_delta, k=k)


def run(cache_path=CACHE, outdir=OUTDIR):
    """Load a cached fit, run tests A/B/C, write figures to ``outdir`` and return
    a stats dict (used by the report generator / multi-session orchestrator)."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    latents = cache["latents"]
    u_trim = cache["u_trim"]
    p, J = cache["p"], cache["J"]

    pert, unpert = ma.split_perturbed(u_trim)
    onsets = {i: ma.perturbation_onset(u_trim[i]) for i in pert}
    med_onset = int(np.median(list(onsets.values()))) if pert else None
    print(f"p={p}  J={J}  trials: {len(pert)} perturbed, {len(unpert)} unperturbed")
    print(f"latent length T-J = {latents[0].shape[0]}; ELBO {cache['elbo_trace'][-1]:.0f}")
    print(f"onset (trimmed) median: {med_onset}")

    manifold, _unp_train, unp_test = build_manifold(latents, unpert, seed=0)
    k = manifold["k"]
    print(f"unperturbed manifold dimension k = {k} (of p={p})")

    rel = np.arange(-W_PRE, W_POST)
    a = test_A(latents, pert, unp_test, onsets, manifold, rel, outdir=outdir)
    b = test_B(latents, pert, unpert, onsets, manifold, k, J, outdir=outdir)
    c = test_C(cache, latents, pert, onsets, manifold, k, rel, outdir=outdir)
    print(f"\nfigures written to {outdir}/validate_{{A,B,C}}_*.png")
    return dict(p=p, J=J, n_pert=len(pert), n_unp=len(unpert), k=k,
                med_onset=med_onset, elbo=float(cache["elbo_trace"][-1]),
                A=a, B=b, C=c)


def main():
    run(CACHE, OUTDIR)


if __name__ == "__main__":
    main()
