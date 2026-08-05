"""Task B only (participation ratio / effective dimensionality), all sessions.

Recomputes the *new 8-group* participation ratio for every cached session fit and
writes a single combined report + figure.  The 8 groups (see
``run_manifold_analysis.participation_ratio_8groups``):

    unperturbed                | perturbed
    unperturbed reach-pre      | unperturbed reach-post     (reach onset = bin 48)
    perturbed   reach-pre      | perturbed   reach-post
    perturbed   perturb-pre    | perturbed   perturb-post   (per-trial pert onset)

Reach onset is fixed at raw bin 48 (1.0 s) for every trial; perturbation onset is
per-trial.  PR is an equal-N bootstrap (95% CI) within each session; across
sessions we report mean +/- SEM and paired Wilcoxon tests for the before/after
contrasts.

Run:
    /home/sp645/miniconda3/envs/ssm/bin/python run_taskB_all_sessions.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from perturbation_plds import manifold_analysis as ma
import run_manifold_analysis as mani

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "perturbation_plds_results")
SESSIONS = range(0, 8)
# before/after (within-condition) contrasts: (label, post_key, pre_key)
CONTRASTS = [
    ("perturbed: reach post − pre",      "pert_reach_post",   "pert_reach_pre"),
    ("unperturbed: reach post − pre",    "unpert_reach_post", "unpert_reach_pre"),
    ("perturbed: perturb post − pre",    "pert_perturb_post", "pert_perturb_pre"),
]
# perturbed-vs-unperturbed (between-condition) contrasts: (label, A_key, B_key);
# tested as A > B (one-sided) and two-sided, paired across sessions
CONTRASTS_BETWEEN = [
    ("post-reach: perturbed − unperturbed", "pert_reach_post", "unpert_reach_post"),
    ("pre-reach:  perturbed − unperturbed", "pert_reach_pre",  "unpert_reach_pre"),
    ("whole trial: perturbed − unperturbed", "pert",           "unpert"),
]


def collect(sessions=SESSIONS, root=ROOT):
    """Per-session 8-group PR (mean) -> dict key -> list over sessions."""
    means = {key: [] for key, _l, _c in mani.PR_GROUPS}
    used = []
    for s in sessions:
        cache_path = os.path.join(root, f"session{s}", "fit_p10.pkl")
        if not os.path.exists(cache_path):
            print(f"[s{s}] no cache, skipping")
            continue
        cache = pickle.load(open(cache_path, "rb"))
        latents, u_trim, J = cache["latents"], cache["u_trim"], cache["J"]
        pert, unpert = ma.split_perturbed(u_trim)
        onsets = {i: ma.perturbation_onset(u_trim[i]) for i in pert}
        groups, reach, n_eq = mani.participation_ratio_8groups(
            latents, pert, unpert, onsets, J)
        for key in means:
            means[key].append(groups[key][0])
        used.append(s)
        print(f"[s{s}] J={J} reach@{reach} n_eq={n_eq}  "
              + "  ".join(f"{k}={groups[k][0]:.2f}" for k, _l, _c in mani.PR_GROUPS))
    return means, used


def make_figure(means, used, out_png):
    keys = [k for k, _l, _c in mani.PR_GROUPS]
    labels = [l for _k, l, _c in mani.PR_GROUPS]
    colors = [c for _k, _l, c in mani.PR_GROUPS]
    xs = np.arange(len(keys))
    M = np.array([means[k] for k in keys])             # (8 groups, n_sessions)
    mean = M.mean(axis=1)
    sem = M.std(axis=1, ddof=1) / np.sqrt(M.shape[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, mean, yerr=sem, color=colors, capsize=3, alpha=0.85, zorder=1)
    # per-session points
    rng = np.random.default_rng(0)
    for gi in range(len(keys)):
        jit = (rng.random(M.shape[1]) - 0.5) * 0.28
        ax.scatter(xs[gi] + jit, M[gi], s=14, color="k", alpha=0.55, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set(ylabel="participation ratio",
           title=f"Effective latent dimension — {len(used)} sessions "
                 f"(bar = mean ± SEM, dots = sessions)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nfigure -> {out_png}")
    return mean, sem


def write_report(means, used, mean, sem, out_md, fig_name):
    keys = [k for k, _l, _c in mani.PR_GROUPS]
    labels = {k: l for k, l, _c in mani.PR_GROUPS}

    # per-session table
    head = "| session | " + " | ".join(labels[k] for k in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    body = []
    for si, s in enumerate(used):
        body.append("| " + str(s) + " | "
                    + " | ".join(f"{means[k][si]:.2f}" for k in keys) + " |")
    avg = ("| **mean±SEM** | "
           + " | ".join(f"{mean[gi]:.2f}±{sem[gi]:.2f}" for gi in range(len(keys)))
           + " |")

    pstr = lambda p: f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"

    # across-session paired contrasts (within-condition before/after)
    crows = []
    for lbl, post_k, pre_k in CONTRASTS:
        post = np.array(means[post_k]); pre = np.array(means[pre_k])
        diff = post - pre
        _, p_g = wilcoxon(post, pre, alternative="greater")
        _, p_2 = wilcoxon(post, pre)  # two-sided
        crows.append(f"| {lbl} | {pre.mean():.2f} | {post.mean():.2f} | "
                     f"{diff.mean():+.2f} | {pstr(p_g)} | {pstr(p_2)} |")

    # across-session paired contrasts (perturbed vs unperturbed)
    brows = []
    for lbl, a_k, b_k in CONTRASTS_BETWEEN:
        a = np.array(means[a_k]); b = np.array(means[b_k])
        diff = a - b
        _, p_g = wilcoxon(a, b, alternative="greater")
        _, p_2 = wilcoxon(a, b)  # two-sided
        brows.append(f"| {lbl} | {b.mean():.2f} | {a.mean():.2f} | "
                     f"{diff.mean():+.2f} | {pstr(p_g)} | {pstr(p_2)} |")

    # difference-of-differences: does the perturbation add extra reach expansion?
    dd_pert = np.array(means["pert_reach_post"]) - np.array(means["pert_reach_pre"])
    dd_unp = np.array(means["unpert_reach_post"]) - np.array(means["unpert_reach_pre"])
    _, p_dd_g = wilcoxon(dd_pert, dd_unp, alternative="greater")
    _, p_dd_2 = wilcoxon(dd_pert, dd_unp)

    lines = [
        "# Task B — effective dimensionality (participation ratio), all sessions",
        "",
        "*Auto-generated by `run_taskB_all_sessions.py`.*",
        "",
        "## Groups (8)",
        "- **reach onset** = raw bin 48 (1.0 s), the *same frame for every trial* "
        "(latent index 48 − J); splits both unperturbed and perturbed trials.",
        "- **perturbation onset** = per-trial first bin with u = 1 (perturbed only).",
        "- PR = `(Σλ)²/Σλ²` of the latent covariance, equal-N trial bootstrap "
        "within each session; across sessions: mean ± SEM and paired Wilcoxon.",
        "",
        f"![taskB summary]({fig_name})",
        "",
        "## Per-session participation ratio",
        "",
        head, sep, *body, avg,
        "",
        "## Before/after contrasts across sessions (paired Wilcoxon, n = "
        f"{len(used)})",
        "",
        "| contrast | pre (mean) | post (mean) | Δ | p (post>pre) | p (two-sided) |",
        "|---|---|---|---|---|---|",
        *crows,
        "",
        "## Perturbed vs unperturbed across sessions (paired Wilcoxon, n = "
        f"{len(used)})",
        "",
        "| contrast | unpert (mean) | pert (mean) | Δ | p (pert>unpert) | p (two-sided) |",
        "|---|---|---|---|---|---|",
        *brows,
        "",
        "## Reach expansion: perturbed vs unperturbed (difference-of-differences)",
        "",
        "Per-session reach effect (post − pre), perturbed vs unperturbed; paired "
        f"Wilcoxon, n = {len(used)}. Tests whether the perturbation adds reach "
        "expansion *beyond* the reach itself.",
        "",
        "| quantity | mean | p (pert>unpert) | p (two-sided) |",
        "|---|---|---|---|",
        f"| perturbed reach (post − pre) | {dd_pert.mean():+.2f} | | |",
        f"| unperturbed reach (post − pre) | {dd_unp.mean():+.2f} | | |",
        f"| **difference (pert − unpert)** | {(dd_pert - dd_unp).mean():+.2f} | "
        f"**{pstr(p_dd_g)}** | {pstr(p_dd_2)} |",
        "",
        "Session-wise reach effect and the paired difference:",
        "",
        "| session | " + " | ".join(str(s) for s in used) + " |",
        "|" + "---|" * (len(used) + 1),
        "| perturbed (post−pre) | "
        + " | ".join(f"{v:+.2f}" for v in dd_pert) + " |",
        "| unperturbed (post−pre) | "
        + " | ".join(f"{v:+.2f}" for v in dd_unp) + " |",
        "| **difference (pert−unpert)** | "
        + " | ".join(f"{v:+.2f}" for v in (dd_pert - dd_unp)) + " |",
        "",
        "```",
        "difference (pert − unpert) per session = ["
        + ", ".join(f"{v:+.3f}" for v in (dd_pert - dd_unp)) + "]",
        "```",
        "",
        "> The reach split is the same frame for all trials; the perturbation "
        "split is per-trial. Comparing the two reveals whether the dimensionality "
        "change is tied to the reach itself or specifically to the perturbation.",
        "",
    ]
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report -> {out_md}")


def main():
    means, used = collect()
    if not used:
        raise RuntimeError("no caches found under " + ROOT)
    fig_name = "all_sessions_taskB_participation_ratio.png"
    out_png = os.path.join(ROOT, fig_name)
    out_md = os.path.join(ROOT, "all_sessions_taskB_report.md")
    mean, sem = make_figure(means, used, out_png)
    write_report(means, used, mean, sem, out_md, fig_name)


if __name__ == "__main__":
    main()
