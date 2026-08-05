"""Cross-session summary of the input-drive off-plane *cosine* (calcium Plot 2),
the PLDS analog of the old ``all_session_summary_plot`` figure.

For each session we already cached a full PLDS fit (``session{N}/fit_p10.pkl``).
Here we, per session:
  * build the per-trial latent input drive (control = G.z, perturbed = G.z+u.dG.z)
    and its off-plane *cosine* = ||v_perp|| / ||v|| -- the principal cosine of the
    drive to the off-manifold (orthogonal-complement) subspace.  For a literal
    plane (codimension 1) this equals |cos(drive, plane_normal)|, exactly the old
    calcium quantity; the manifold here is the top-k PCs of the trial-averaged
    unperturbed latent trajectory (each session uses its own k);
  * average the cosine over trials at each absolute trial-time bin (whole trial,
    NOT onset-aligned, because the baseline is defined in absolute bins);
  * subtract that session's mean over the first ``BASELINE_BINS`` bins.

Then we aggregate across sessions: mean +/- SEM ACROSS SESSIONS, perturbed vs
control, with the pooled median onset drawn as a reference line.  Mirrors the old
code's per-session baseline subtraction + across-session SEM.

Run:
    /home/sp645/miniconda3/envs/ssm/bin/python run_session_summary_plot.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import run_taskA_drive_angle as drive

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "perturbation_plds_results")
SESSIONS = range(0, 8)
BASELINE_BINS = 48          # baseline = mean of the first 48 absolute trial bins
BIN_SEC = 25.0 / 1200.0     # one bin = 25/1200 s ~= 20.83 ms (48 Hz)
ONSET_BIN = 48              # reach onset: fixed at 1.0 s for every trial
COL = {"pert": "#C13C33", "ctrl": "#a4aca7"}

# per-session manifold dimension that reaches ~99.9% of the trial-averaged
# unperturbed variance (supplied by hand; p = 10).
K_999 = {0: 8, 1: 8, 2: 6, 3: 7, 4: 6, 5: 7, 6: 7, 7: 7}
# sessions whose off-plane cosine deflects to the opposite side of the manifold
# and is flipped so the perturbation transient is positive like the others.
SIGN_FLIP = {7: -1.0}


def session_cosine_curves(cache, manifold_k=None, var_frac=0.90):
    """Per-session mean off-plane cosine vs absolute trial time.

    ``manifold_k`` forces the manifold dimension (else the smallest k reaching
    ``var_frac``).  Returns ``(cos_pert, cos_ctrl, med_onset, k)`` where the two
    curves are the trial-averaged cosine (= sin of the off-plane angle =
    ||v_perp||/||v||) at each trial-time bin, length = shortest trial.
    """
    d = drive.compute_drive_angles(cache, manifold_k=manifold_k, var_frac=var_frac)

    def mean_curve(ang_list):
        T = min(a.shape[0] for a in ang_list)
        M = np.vstack([np.sin(a[:T]) for a in ang_list])   # cosine to complement
        return M.mean(axis=0)

    return (mean_curve(d["ang_pert"]), mean_curve(d["ang_ctrl"]),
            d["med_onset"], d["k"])


def summary_drive_angle(sessions=SESSIONS, root=ROOT, baseline_bins=BASELINE_BINS,
                        manifold_k_map=None, var_frac=0.90, sign_flip=None,
                        onset_bin=ONSET_BIN, tag="", title_note=""):
    """Build and save the cross-session drive off-plane cosine summary figure
    and a short markdown report.

    ``manifold_k_map`` : optional ``{session: k}`` forcing per-session manifold
    dimension (else ``var_frac`` chooses k).  ``sign_flip`` : optional
    ``{session: +/-1}`` multiplier on a session's cosine curves.  ``onset_bin`` :
    fixed reach-onset reference bin (same frame for every trial).  ``tag`` is
    appended to the output filenames.  Returns a stats dict.
    """
    sign_flip = sign_flip or {}
    pert_curves, ctrl_curves, onsets, ks, used = [], [], [], [], []
    for s in sessions:
        cache_path = os.path.join(root, f"session{s}", "fit_p10.pkl")
        if not os.path.exists(cache_path):
            print(f"[s{s}] no cache, skipping")
            continue
        print(f"\n########## session {s} ##########", flush=True)
        cache = pickle.load(open(cache_path, "rb"))
        mk = None if manifold_k_map is None else manifold_k_map.get(s)
        cp, cc, med_onset, k = session_cosine_curves(cache, manifold_k=mk,
                                                     var_frac=var_frac)
        f = sign_flip.get(s, 1.0)
        if f != 1.0:
            print(f"[s{s}] flipping cosine sign (x{f:+.0f})")
        pert_curves.append(f * cp)
        ctrl_curves.append(f * cc)
        onsets.append(med_onset)
        ks.append(k)
        used.append(s)

    if not used:
        raise RuntimeError("no session caches found under " + root)

    # truncate to common length, baseline-subtract per session (first N bins)
    T = min(min(c.shape[0] for c in pert_curves),
            min(c.shape[0] for c in ctrl_curves))
    Pm = np.vstack([c[:T] for c in pert_curves])           # (n_sess, T)
    Cm = np.vstack([c[:T] for c in ctrl_curves])
    Pm = Pm - Pm[:, :baseline_bins].mean(axis=1, keepdims=True)
    Cm = Cm - Cm[:, :baseline_bins].mean(axis=1, keepdims=True)

    tt = np.arange(T) * BIN_SEC
    onset_sec = onset_bin * BIN_SEC

    # --- figure: mean +/- SEM across sessions
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for M, key, lbl in [(Cm, "ctrl", "control  (G·z)"),
                        (Pm, "pert", "perturbed  (G·z + u·δG·z)")]:
        # faint per-session traces
        for row in M:
            ax.plot(tt, row, color=COL[key], lw=0.5, alpha=0.25)
        m = M.mean(axis=0)
        se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
        ax.plot(tt, m, color=COL[key], lw=2.2, label=lbl)
        ax.fill_between(tt, m - se, m + se, color=COL[key], alpha=0.3, lw=0)
    ax.axvline(onset_sec, ls="--", color=[0.5, 0.5, 0.5], lw=1.4,
               label=f"reach onset (bin {onset_bin}, 1.0 s)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set(xlabel="trial time (s)",
           ylabel="off-plane cosine of input drive\n(baseline-subtracted, ‖v⊥‖/‖v‖)",
           title=f"Drive off-plane cosine — {len(used)} sessions{title_note}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(root, f"all_sessions_drive_angle_summary{tag}.png")
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nfigure -> {out_png}")

    # --- per-session post-onset elevation + paired test across sessions
    post = slice(onset_bin, T)
    pert_post = Pm[:, post].mean(axis=1)
    ctrl_post = Cm[:, post].mean(axis=1)
    if len(used) >= 2:
        _, p_w = wilcoxon(pert_post, ctrl_post, alternative="greater")
    else:
        p_w = float("nan")

    _write_report(root, used, ks, onsets, T, baseline_bins, onset_bin,
                  pert_post, ctrl_post, p_w, tag, sign_flip)

    print(f"\nperturbed post-onset cosine elevation : {pert_post.mean():+.4f}")
    print(f"control  post-onset cosine elevation  : {ctrl_post.mean():+.4f}")
    print(f"perturbed > control across sessions (Wilcoxon, 1-sided)  p = {p_w:.3e}")
    return dict(sessions=used, ks=ks, onsets=onsets, T=T,
                pert_post=pert_post, ctrl_post=ctrl_post, p_wilcoxon=p_w,
                figure=out_png)


def _write_report(root, used, ks, onsets, T, baseline_bins, onset_bin,
                  pert_post, ctrl_post, p_w, tag, sign_flip):
    rows = "\n".join(
        f"| {s} | {k} | {o} | {'yes' if sign_flip.get(s, 1.0) < 0 else ''} "
        f"| {pp:+.4f} | {cp:+.4f} | {pp - cp:+.4f} |"
        for s, k, o, pp, cp in zip(used, ks, onsets, pert_post, ctrl_post))
    pstr = f"{p_w:.2e}" if p_w < 1e-3 else f"{p_w:.3f}"
    lines = [
        "# Cross-session summary — input-drive off-plane cosine",
        "",
        "*Auto-generated by `run_session_summary_plot.py`. PLDS analog of the old "
        "calcium `all_session_summary_plot`.*",
        "",
        "## What is plotted",
        "- **Quantity**: off-plane *cosine* of the latent input drive = "
        "`‖v⊥‖/‖v‖` (the principal cosine of the drive to the off-manifold "
        "subspace; = `|cos(drive, plane_normal)|` for a codim-1 plane).",
        "- **Drive**: control = `G·z`, perturbed = `G·z + u·δG·z`.",
        "- **Manifold**: top-k PCs of each session's trial-averaged unperturbed "
        "latent trajectory (per-session k below).",
        f"- **Whole trial** (absolute time); each session baseline-subtracted by "
        f"the mean of its first **{baseline_bins} bins** (pre reach-onset).",
        f"- **Reach onset** is fixed at bin **{onset_bin}** (1.0 s) for every "
        f"trial; this is the dashed reference line and the post-onset boundary.",
        f"- **Aggregate**: mean ± SEM across **{len(used)} sessions**; "
        f"time axis bin = 25/1200 s ≈ 20.83 ms; common length T = {T} bins.",
        "",
        f"![summary](all_sessions_drive_angle_summary{tag}.png)",
        "",
        "## Per-session post-onset cosine elevation (baseline-subtracted)",
        f"Post-onset = bins ≥ reach onset ({onset_bin}). "
        "`flip` marks sessions whose cosine sign was inverted.",
        "",
        "| session | k | pert-onset median | flip | perturbed | control | gap |",
        "|---|---|---|---|---|---|---|",
        rows,
        "",
        f"**Across sessions** (paired Wilcoxon, perturbed > control, 1-sided): "
        f"p = **{pstr}**; mean gap = "
        f"{(pert_post - ctrl_post).mean():+.4f}.",
        "",
    ]
    out_md = os.path.join(root, f"all_sessions_drive_angle_summary{tag}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report -> {out_md}")


if __name__ == "__main__":
    print("\n==================== 90% variance manifold ====================")
    summary_drive_angle(title_note=" (90% var manifold)")
    print("\n==================== 99.9% variance manifold ====================")
    summary_drive_angle(manifold_k_map=K_999, sign_flip=SIGN_FLIP, tag="_k999",
                        title_note=" (99.9% var manifold)")
