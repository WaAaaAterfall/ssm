"""Task A, "off-plane angle" version -- PLDS port of the old calcium Plot 2.

Old method (calcium + ridge), Plot 2 "Off-plane angle":
  * fit a 2-D plane to the trial-averaged CONTROL neural trajectory (in PCA-3);
  * per trial/time build the instantaneous FIR drive into neural space,
        control drive  = K0.z          (= G.z)
        perturbed drive= K0.z + u_t K1.z (= G.z + u_t dG.z),  per-lag gated;
  * report the angle of that drive off the plane = 90 - arccos(|cos(drive, normal)|).

PLDS port (this script):
  * the latent s in R^p already IS the denoised low-D space (no neural PCA);
  * the "plane" -> the manifold spanned by the trial-averaged UNPERTURBED latent
    trajectory (top-k PCs); a single normal -> the (p-k)-D orthogonal complement;
  * drive lives in the LATENT (G, dG map kinematics -> latent), computed by
    ma.realized_drive (per-lag, u-gated for dG);
  * angle off plane = degrees(arcsin(||drive_perp|| / ||drive||)), which equals
    the old 90 - arccos(...) when k = p-1 (a true plane).

Convention: u==1 marks perturbed bins (the current data convention).

Run:
    /home/sp645/miniconda3/envs/ssm/bin/python run_taskA_drive_angle.py
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from perturbation_plds import manifold_analysis as ma

CACHE = "perturbation_plds_results/session0/fit_p10.pkl"
OUTDIR = "perturbation_plds_results/session0"
W_PRE, W_POST = 20, 60
REACH_ONSET = 48       # reach onset (1.0 s), same trial-time bin for every trial
MANIFOLD_K = None      # None -> 90% variance of the mean unperturbed trajectory;
                       # set e.g. 2 or 3 to force a plane-like low-D manifold.


def compute_drive_angles(cache, manifold_k=MANIFOLD_K, var_frac=0.90):
    """Per-trial latent input-drive vectors and their angle off the unperturbed
    manifold, for one cached fit.

    Shared by :func:`run` and the cross-session summary so both use exactly the
    same drive construction (control = G.z, perturbed = G.z + u.dG.z), the same
    reference manifold (top-k PCs of the trial-averaged unperturbed latent
    trajectory), and the same off-plane angle.  Returns a dict with the raw drive
    vectors, the per-trial angle time series (radians, absolute trial time), the
    trial index lists, onsets, the manifold basis and ``k``.
    """
    latents = cache["latents"]
    u_trim = cache["u_trim"]
    G_lag, dG_lag = cache["G_lag"], cache["dG_lag"]
    Z_full, U_full = cache["Z_full"], cache["U_full"]
    p = cache["p"]

    pert, unpert = ma.split_perturbed(u_trim)
    onsets = {i: ma.perturbation_onset(u_trim[i]) for i in pert}
    med_onset = int(np.median(list(onsets.values()))) if pert else None
    print(f"p={p}  perturbed {len(pert)}  unperturbed {len(unpert)}  median onset {med_onset}")

    # --- manifold = top-k PCs of the trial-averaged UNPERTURBED latent trajectory
    L = min(latents[i].shape[0] for i in unpert)
    mean_unp_traj = np.mean([latents[i][:L] for i in unpert], axis=0)   # (L, p)
    manifold = ma.fit_manifold([mean_unp_traj], k=manifold_k, var_frac=var_frac)
    basis, k = manifold["basis"], manifold["k"]
    print(f"manifold (mean unperturbed trajectory): k = {k} of p = {p}  "
          f"(orthogonal complement is {p - k}-D)")

    # --- per-trial drive vectors in the latent, and their angle off the manifold
    def perturbed_drive(i):
        g = ma.realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="G")
        dg = ma.realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="dG")
        return g + dg                                  # K0.z + u_t K1.z
    def control_drive(i):
        return ma.realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="G")  # K0.z

    pdrives = [perturbed_drive(i) for i in pert]
    cdrives = [control_drive(i) for i in unpert]
    ang_pert = [ma.direction_off_manifold_angle(d, basis) for d in pdrives]
    ang_ctrl = [ma.direction_off_manifold_angle(d, basis) for d in cdrives]

    return dict(pdrives=pdrives, cdrives=cdrives, ang_pert=ang_pert,
                ang_ctrl=ang_ctrl, pert=pert, unpert=unpert, onsets=onsets,
                med_onset=med_onset, basis=basis, k=k, p=p,
                mean_unp_traj=mean_unp_traj)


def run(cache_path=CACHE, outdir=OUTDIR):
    """Drive-angle (calcium Plot 2) port for one cached fit.  Writes the four
    figures to ``outdir`` and returns a stats dict."""
    cache = pickle.load(open(cache_path, "rb"))
    d = compute_drive_angles(cache)
    pdrives, cdrives = d["pdrives"], d["cdrives"]
    ang_pert, ang_ctrl = d["ang_pert"], d["ang_ctrl"]
    pert, unpert = d["pert"], d["unpert"]
    onsets, med_onset = d["onsets"], d["med_onset"]
    k, p, mean_unp_traj = d["k"], d["p"], d["mean_unp_traj"]

    aligned = plot_onset_aligned(ang_pert, ang_ctrl, pert, onsets, med_onset, k, outdir)
    whole = plot_whole_trial(ang_pert, ang_ctrl, med_onset, k, outdir)
    sweep = plot_k_sweep(pdrives, cdrives, mean_unp_traj, outdir=outdir)
    plot_whole_trial_ksweep(pdrives, cdrives, mean_unp_traj, med_onset, outdir=outdir)
    return dict(p=p, k=k, n_pert=len(pert), n_unp=len(unpert), med_onset=med_onset,
                aligned=aligned, whole=whole, ksweep=sweep)


def main():
    run(CACHE, OUTDIR)


def plot_onset_aligned(ang_pert, ang_ctrl, pert, onsets, med_onset, k, outdir=OUTDIR):
    """Original version: per-trial onset-aligned (onset varies, bin 27-60)."""
    rel = np.arange(-W_PRE, W_POST)

    def aligned_matrix(ang_list, onset_of):
        M = np.full((len(ang_list), len(rel)), np.nan)
        for r, a in enumerate(ang_list):
            o = onset_of(r)
            for j, dd in enumerate(rel):
                if 0 <= o + dd < a.shape[0]:
                    M[r, j] = np.degrees(a[o + dd])
        return M

    Mp = aligned_matrix(ang_pert, lambda r: onsets[pert[r]])
    Mc = aligned_matrix(ang_ctrl, lambda _r: med_onset)

    fig, ax = plt.subplots(figsize=(6, 5))
    for M, lbl, c in [(Mc, "control (G·z)", "#a4aca7"),
                      (Mp, "perturbed (G·z + u·δG·z)", "#C13C33")]:
        m = np.nanmean(M, 0); se = np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
        ax.plot(rel, m, color=c, lw=1.8, label=lbl)
        ax.fill_between(rel, m - se, m + se, color=c, alpha=.2)
    ax.axvline(0, ls="--", c="k", lw=1, label="perturbation onset")
    ax.set(xlabel="bins from onset", ylabel="drive angle off plane (deg)  [90−arccos]",
           title=f"Off-plane angle of the input drive (onset-aligned, k={k})")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"{outdir}/validate_A_drive_angle.png"
    plt.savefig(out, dpi=130); plt.close(fig)

    post = np.array([np.nanmean(ang_pert[r][onsets[i]:onsets[i] + W_POST])
                     for r, i in enumerate(pert)]) * 180 / np.pi
    pre = np.array([np.nanmean(ang_pert[r][max(0, onsets[i] - W_PRE):onsets[i]])
                    for r, i in enumerate(pert)]) * 180 / np.pi
    ctrl = np.array([np.degrees(a).mean() for a in ang_ctrl])
    _, p_mw = mannwhitneyu(post, ctrl, alternative="greater")
    _, p_w = wilcoxon(post, pre, alternative="greater")
    print(f"\n[onset-aligned] perturbed post-onset : {post.mean():.2f} +/- {post.std():.2f} deg")
    print(f"[onset-aligned] control              : {ctrl.mean():.2f} +/- {ctrl.std():.2f} deg")
    print(f"  perturbed > control (Mann-Whitney, 1-sided)  p = {p_mw:.2e}")
    print(f"[onset-aligned] perturbed pre-onset  : {pre.mean():.2f} +/- {pre.std():.2f} deg")
    print(f"  post > pre within trial (Wilcoxon, 1-sided)  p = {p_w:.2e}")
    print(f"  figure -> {out}")
    return dict(post=post.mean(), post_sd=post.std(), ctrl=ctrl.mean(),
                ctrl_sd=ctrl.std(), pre=pre.mean(), pre_sd=pre.std(),
                p_pert_vs_ctrl=p_mw, p_post_vs_pre=p_w)


def plot_whole_trial(ang_pert, ang_ctrl, med_onset, k, outdir=OUTDIR):
    """New version: absolute trial time, whole trial, NO onset alignment -- like
    the old calcium version (whose onset was at a fixed frame).  All trials are
    averaged at each trial-time bin; median onset drawn only as a reference."""
    def trialtime_matrix(ang_list):
        T = min(a.shape[0] for a in ang_list)
        return np.vstack([np.degrees(a[:T]) for a in ang_list])     # (n_trials, T)

    Mp, Mc = trialtime_matrix(ang_pert), trialtime_matrix(ang_ctrl)
    Tcommon = min(Mp.shape[1], Mc.shape[1])
    Mp, Mc = Mp[:, :Tcommon], Mc[:, :Tcommon]
    tt = np.arange(Tcommon)

    fig, ax = plt.subplots(figsize=(6, 5))
    for M, lbl, c in [(Mc, "control (G·z)", "#a4aca7"),
                      (Mp, "perturbed (G·z + u·δG·z)", "#C13C33")]:
        m = M.mean(0); se = M.std(0) / np.sqrt(M.shape[0])
        ax.plot(tt, m, color=c, lw=1.8, label=lbl)
        ax.fill_between(tt, m - se, m + se, color=c, alpha=.2)
    ax.axvline(REACH_ONSET, ls="--", c="k", lw=1, label=f"reach onset (bin {REACH_ONSET})")
    ax.set(xlabel="trial time (bin)", ylabel="drive angle off plane (deg)  [90−arccos]",
           title=f"Off-plane angle of the input drive (whole trial, k={k})")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"{outdir}/validate_A_drive_angle_wholetrial.png"
    plt.savefig(out, dpi=130); plt.close(fig)

    pert_mean = np.array([np.degrees(a).mean() for a in ang_pert])
    ctrl_mean = np.array([np.degrees(a).mean() for a in ang_ctrl])
    _, p_mw = mannwhitneyu(pert_mean, ctrl_mean, alternative="greater")
    print(f"\n[whole-trial] perturbed : {pert_mean.mean():.2f} +/- {pert_mean.std():.2f} deg")
    print(f"[whole-trial] control   : {ctrl_mean.mean():.2f} +/- {ctrl_mean.std():.2f} deg")
    print(f"  perturbed > control (Mann-Whitney, 1-sided)  p = {p_mw:.2e}")
    print(f"  figure -> {out}")
    return dict(pert=pert_mean.mean(), pert_sd=pert_mean.std(),
                ctrl=ctrl_mean.mean(), ctrl_sd=ctrl_mean.std(), p_pert_vs_ctrl=p_mw)


def plot_whole_trial_ksweep(pdrives, cdrives, mean_traj, med_onset, ks=range(2, 10), outdir=OUTDIR):
    """Whole-trial drive-angle time course (control vs perturbed vs trial time),
    one panel per manifold dimension k -- the `validate_A_drive_angle_wholetrial`
    plot repeated across k."""
    ks = list(ks)
    ncol = 4
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                             sharex=True, squeeze=False)
    for ax, K in zip(axes.ravel(), ks):
        b = ma.fit_manifold([mean_traj], k=K)["basis"]
        ang_p = [ma.direction_off_manifold_angle(d, b) for d in pdrives]
        ang_c = [ma.direction_off_manifold_angle(d, b) for d in cdrives]
        T = min(min(a.shape[0] for a in ang_p), min(a.shape[0] for a in ang_c))
        Mp = np.vstack([np.degrees(a[:T]) for a in ang_p])
        Mc = np.vstack([np.degrees(a[:T]) for a in ang_c])
        tt = np.arange(T)
        for M, c in [(Mc, "#a4aca7"), (Mp, "#C13C33")]:
            m = M.mean(0); se = M.std(0) / np.sqrt(M.shape[0])
            ax.plot(tt, m, color=c, lw=1.5)
            ax.fill_between(tt, m - se, m + se, color=c, alpha=.2)
        ax.axvline(REACH_ONSET, ls="--", c="k", lw=1)
        ax.set_title(f"k = {K}")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[len(ks):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("trial time (bin)")
    for ax in axes[:, 0]:
        ax.set_ylabel("drive angle off plane (deg)")
    fig.legend(["control (G·z)", "perturbed (G·z + u·δG·z)", f"reach onset (bin {REACH_ONSET})"],
               loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{outdir}/validate_A_drive_angle_wholetrial_ksweep.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"\n[whole-trial k-sweep panels] figure -> {out}")


def plot_k_sweep(pdrives, cdrives, mean_traj, ks=range(2, 10), outdir=OUTDIR):
    """Whole-trial control vs perturbed drive angle as the manifold dimension k
    grows.  As k increases the manifold absorbs more of the baseline G.z drive,
    so the control (baseline) angle shrinks; the perturbed>control gap shows
    whether the perturbed excess survives a larger reference subspace."""
    ks = list(ks)
    mc = mean_traj - mean_traj.mean(0)
    cum = np.cumsum(np.linalg.svd(mc, compute_uv=False) ** 2)
    cum = cum / cum[-1]
    cvals, pvals = [], []
    for K in ks:
        b = ma.fit_manifold([mean_traj], k=K)["basis"]
        cvals.append(np.mean([np.degrees(ma.direction_off_manifold_angle(d, b)).mean() for d in cdrives]))
        pvals.append(np.mean([np.degrees(ma.direction_off_manifold_angle(d, b)).mean() for d in pdrives]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ks, cvals, "-o", color="#a4aca7", lw=1.8, label="control (G·z)")
    ax.plot(ks, pvals, "-o", color="#C13C33", lw=1.8, label="perturbed (G·z + u·δG·z)")
    ax.set(xlabel="manifold dimension k", ylabel="whole-trial drive angle off plane (deg)",
           title="Off-plane drive angle vs manifold dimension")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"{outdir}/validate_A_drive_angle_ksweep.png"
    plt.savefig(out, dpi=130); plt.close(fig)

    print("\n[k-sweep, whole trial]  k : var%  control  perturbed  gap")
    for K, cv, pv, cm in zip(ks, cvals, pvals, cum[np.array(ks) - 1]):
        print(f"   {K} : {100 * cm:4.1f}%   {cv:5.2f}    {pv:5.2f}    {pv - cv:+.2f}")
    print(f"  figure -> {out}")
    return dict(ks=ks, var=[float(cum[K - 1]) for K in ks], control=cvals,
                perturbed=pvals)


if __name__ == "__main__":
    main()
