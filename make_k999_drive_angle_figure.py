"""Standalone regeneration of ``all_sessions_drive_angle_summary_k999.png``.

This is the code path behind that figure, collapsed into one file: the
``summary_drive_angle(manifold_k_map=K_999, sign_flip=SIGN_FLIP, tag="_k999")``
call at the bottom of ``run_session_summary_plot.py``, plus the two pieces it
imports -- ``run_taskA_drive_angle.compute_drive_angles`` and the primitives it
uses from ``perturbation_plds.manifold_analysis`` (``fit_manifold``,
``realized_drive``, ``direction_off_manifold_angle``, ``split_perturbed``,
``perturbation_onset``).  Nothing here imports the project package, so it runs
against the cached fits alone.

What the figure shows, per session:
  * the per-trial latent input drive -- control = ``G·z`` on unperturbed trials,
    perturbed = ``G·z + u·δG·z`` on perturbed trials;
  * its off-plane *cosine* ``‖v⊥‖/‖v‖``, the principal cosine of that drive to
    the orthogonal complement of the unperturbed manifold (= ``|cos(drive,
    normal)|`` when the manifold is codimension 1, i.e. the old calcium
    quantity);
  * the manifold is the top-``k`` PCs of the trial-averaged *unperturbed* latent
    trajectory, with ``k`` fixed per session by ``K_999`` (~99.9% of that
    trajectory's variance) rather than chosen by a variance target;
  * trial-averaged at each absolute trial-time bin (whole trial, NOT
    onset-aligned, because the baseline is defined in absolute bins), then
    baseline-subtracted by the mean of the first ``BASELINE_BINS`` bins.

Across sessions: mean ± SEM over the 8 session curves, faint per-session traces
underneath, and a paired 1-sided Wilcoxon on the post-onset elevation.

Run:
    /home/sp645/miniconda3/envs/ssm/bin/python make_k999_drive_angle_figure.py

By default this overwrites the committed
``perturbation_plds_results/all_sessions_drive_angle_summary_k999.png``;
pass ``--out`` to write elsewhere.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

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


# --------------------------------------------------------------------------- #
# primitives (from perturbation_plds.manifold_analysis)
# --------------------------------------------------------------------------- #
def fit_manifold(latents, k=None, var_frac=0.90):
    """PCA subspace of a set of latent trajectories.

    Returns ``mean`` (p,), ``basis`` (p, k) with orthonormal columns, ``eigvals``
    (p,) descending, and ``k``.  With ``k=None`` the smallest k reaching
    ``var_frac`` of the variance is used.
    """
    X = np.concatenate([np.asarray(s) for s in latents], axis=0)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD of centered data; eigenvalues of covariance = s**2 / (n-1)
    _U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (s ** 2) / max(Xc.shape[0] - 1, 1)
    if k is None:
        cum = np.cumsum(eigvals) / np.sum(eigvals)
        k = int(np.searchsorted(cum, var_frac) + 1)
    k = max(1, min(k, Vt.shape[0]))
    return dict(mean=mean, basis=Vt[:k].T, eigvals=eigvals, k=k)


def direction_off_manifold_angle(vectors, basis, eps=1e-9):
    """Angle of each row *direction* off the manifold subspace (NO centering).

    ``arcsin(‖v⊥‖/‖v‖)`` with ``v⊥ = v - basis (basis^T v)``.  The manifold mean
    is deliberately not subtracted: the drive is a direction through the origin,
    exactly as the old calcium code dotted the *normalized* drive with the plane
    normal and ignored the plane offset.  ``vectors`` : (T, p) -> (T,) radians.
    """
    basis = np.asarray(basis)
    P = basis @ basis.T
    v = np.asarray(vectors, dtype=float)
    v_perp = v - v @ P.T
    norm = np.linalg.norm(v, axis=1)
    ratio = np.linalg.norm(v_perp, axis=1) / np.clip(norm, eps, None)
    return np.arcsin(np.clip(ratio, 0.0, 1.0))


def realized_drive(G_lag, dG_lag, z_full, u_full, which="dG"):
    """Instantaneous input drive into the latent at each trimmed time step.

    ``which='dG'`` -> sum_l u_{t-l} dG_l z_{t-l} (perturbation-only drive);
    ``which='G'``  -> sum_l G_l z_{t-l} (baseline drive).  Returns ``(T, p)``.
    """
    z = np.asarray(z_full, dtype=float)
    u = np.asarray(u_full, dtype=float).reshape(-1)
    Jp1, p, _d_z = dG_lag.shape
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


# --------------------------------------------------------------------------- #
# per-session curves (from run_taskA_drive_angle + run_session_summary_plot)
# --------------------------------------------------------------------------- #
def compute_drive_angles(cache, manifold_k=None, var_frac=0.90):
    """Per-trial latent input-drive vectors and their angle off the unperturbed
    manifold, for one cached fit."""
    latents = cache["latents"]
    u_trim = cache["u_trim"]
    G_lag, dG_lag = cache["G_lag"], cache["dG_lag"]
    Z_full, U_full = cache["Z_full"], cache["U_full"]
    p = cache["p"]

    pert, unpert = split_perturbed(u_trim)
    onsets = {i: perturbation_onset(u_trim[i]) for i in pert}
    med_onset = int(np.median(list(onsets.values()))) if pert else None
    print(f"p={p}  perturbed {len(pert)}  unperturbed {len(unpert)}  "
          f"median onset {med_onset}")

    # --- manifold = top-k PCs of the trial-averaged UNPERTURBED latent trajectory
    L = min(latents[i].shape[0] for i in unpert)
    mean_unp_traj = np.mean([latents[i][:L] for i in unpert], axis=0)   # (L, p)
    manifold = fit_manifold([mean_unp_traj], k=manifold_k, var_frac=var_frac)
    basis, k = manifold["basis"], manifold["k"]
    print(f"manifold (mean unperturbed trajectory): k = {k} of p = {p}  "
          f"(orthogonal complement is {p - k}-D)")

    # --- per-trial drive vectors in the latent, and their angle off the manifold
    pdrives = [realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="G")
               + realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="dG")
               for i in pert]                       # K0.z + u_t K1.z
    cdrives = [realized_drive(G_lag, dG_lag, Z_full[i], U_full[i], which="G")
               for i in unpert]                     # K0.z
    ang_pert = [direction_off_manifold_angle(d, basis) for d in pdrives]
    ang_ctrl = [direction_off_manifold_angle(d, basis) for d in cdrives]

    return dict(ang_pert=ang_pert, ang_ctrl=ang_ctrl, med_onset=med_onset, k=k)


def session_cosine_curves(cache, manifold_k=None, var_frac=0.90):
    """Per-session mean off-plane cosine vs absolute trial time.

    Returns ``(cos_pert, cos_ctrl, med_onset, k)``; the curves are the
    trial-averaged cosine (= sin of the off-plane angle = ‖v⊥‖/‖v‖) at each
    trial-time bin, length = shortest trial.
    """
    d = compute_drive_angles(cache, manifold_k=manifold_k, var_frac=var_frac)

    def mean_curve(ang_list):
        T = min(a.shape[0] for a in ang_list)
        M = np.vstack([np.sin(a[:T]) for a in ang_list])   # cosine to complement
        return M.mean(axis=0)

    return (mean_curve(d["ang_pert"]), mean_curve(d["ang_ctrl"]),
            d["med_onset"], d["k"])


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def make_figure(out_png, sessions=SESSIONS, root=ROOT,
                baseline_bins=BASELINE_BINS, manifold_k_map=K_999,
                sign_flip=SIGN_FLIP, onset_bin=ONSET_BIN,
                title_note=" (99.9% var subspace)"):
    """Build and save the cross-session drive off-plane cosine summary figure.
    Returns a stats dict."""
    sign_flip = sign_flip or {}
    pert_curves, ctrl_curves, onsets, ks, used = [], [], [], [], []
    # One iteration per session: load its cached fit and reduce its ~140 trials
    # to two curves (perturbed, control).  The five lists stay index-aligned --
    # entry j of each describes the j-th session that actually contributed.
    for s in sessions:
        cache_path = os.path.join(root, f"session{s}", "fit_p10.pkl")
        if not os.path.exists(cache_path):
            # missing caches are skipped, not fatal: the figure is built from
            # whatever sessions are present, and `used` records which those were
            print(f"[s{s}] no cache, skipping")
            continue
        print(f"\n########## session {s} ##########", flush=True)
        cache = pickle.load(open(cache_path, "rb"))
        # mk: manifold dimension forced for this session (K_999); None falls back
        # to the smallest k reaching var_frac of the unperturbed variance
        mk = None if manifold_k_map is None else manifold_k_map.get(s)
        # cp / cc: (T_s,) trial-averaged off-plane cosine for this session's
        # perturbed / unperturbed trials.  med_onset: median perturbation-onset
        # bin (reported only -- the dashed line uses the fixed ONSET_BIN, since
        # onset varies trial to trial).  k: manifold dimension actually used.
        cp, cc, med_onset, k = session_cosine_curves(cache, manifold_k=mk)
        # f = -1 for sessions whose effect deflects the opposite way, so the
        # per-session gaps don't cancel when averaged (see SIGN_FLIP)
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

    # T: number of trial-time bins the sessions have in common -- the shortest
    # curve over both conditions and all sessions (165 here, ~3.44 s).  Sessions
    # differ slightly in trial length, so everything is cut to T before stacking.
    T = min(min(c.shape[0] for c in pert_curves),
            min(c.shape[0] for c in ctrl_curves))
    # Pm / Cm: one ROW PER SESSION, one column per trial-time bin -- (n_sess, T).
    # Trials are already gone at this point: row s is session s's average over
    # its ~60 perturbed (Pm) or ~80 unperturbed (Cm) trials, computed in
    # session_cosine_curves.
    Pm = np.vstack([c[:T] for c in pert_curves])           # (n_sess, T)
    Cm = np.vstack([c[:T] for c in ctrl_curves])
    # Baseline: each row minus its OWN mean over bins 0..baseline_bins-1 (the
    # pre-reach-onset window).  keepdims makes the mean an (n_sess, 1) column, so
    # every session gets its own baseline -- and Pm and Cm are baselined
    # independently, i.e. perturbed and unperturbed each keep a separate baseline
    # as in the old regression version.  Nothing is pooled across sessions.
    Pm = Pm - Pm[:, :baseline_bins].mean(axis=1, keepdims=True)
    Cm = Cm - Cm[:, :baseline_bins].mean(axis=1, keepdims=True)

    tt = np.arange(T) * BIN_SEC          # x axis: bin index -> seconds
    onset_sec = onset_bin * BIN_SEC      # dashed reference line (1.0 s)

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
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nfigure -> {out_png}")

    # --- per-session post-onset elevation + paired test across sessions
    post = slice(onset_bin, T)
    pert_post = Pm[:, post].mean(axis=1)
    ctrl_post = Cm[:, post].mean(axis=1)
    p_w = (wilcoxon(pert_post, ctrl_post, alternative="greater")[1]
           if len(used) >= 2 else float("nan"))

    print("\n session | k | onset | perturbed | control  | gap")
    for s, k, o, pp, cp_ in zip(used, ks, onsets, pert_post, ctrl_post):
        print(f"    {s}    | {k} |  {o}  |  {pp:+.4f}  | {cp_:+.4f} | {pp - cp_:+.4f}")
    print(f"\nperturbed post-onset cosine elevation : {pert_post.mean():+.4f}")
    print(f"control  post-onset cosine elevation  : {ctrl_post.mean():+.4f}")
    print(f"mean gap                              : {(pert_post - ctrl_post).mean():+.4f}")
    print(f"perturbed > control across sessions (Wilcoxon, 1-sided)  p = {p_w:.3e}")
    return dict(sessions=used, ks=ks, onsets=onsets, T=T, pert_post=pert_post,
                ctrl_post=ctrl_post, p_wilcoxon=p_w, figure=out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "all_sessions_drive_angle_summary_k999.png"),
        help="output PNG path (default: overwrite the committed figure)")
    ap.add_argument("--root", default=ROOT,
                    help="directory holding session{N}/fit_p10.pkl")
    args = ap.parse_args()
    make_figure(args.out, root=args.root)
