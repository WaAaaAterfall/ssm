#!/usr/bin/env python
"""Run the perturbation-PLDS smoke test on synthetic OR your own data.

This is a thin, swap-the-dataset driver around the package's fitting / evaluation
/ cross-validation routines.  It

  1. loads a dataset (synthetic by default; YOUR data via ``--data``),
  2. fits the full perturbation PLDS,
  3. evaluates it (in-sample Poisson deviance / pseudo-R^2),
  4. cross-validates ``full`` vs ``no_dG`` (and ``intrinsic``) by held-out trial
     deviance -- the actual scientific test of whether ``dG`` is real,
  5. runs the ground-truth sanity checks when ground truth is available
     (synthetic data only), and
  6. saves everything to ``<out>/results.pkl`` for ``validate_results.py``.

------------------------------------------------------------------------------
USING YOUR OWN DATA  (the only thing you need to change)
------------------------------------------------------------------------------
Pass ``--data path/to/file`` where the file is one of:

  *.npz   with arrays ``z``, ``x``, ``u``.  Either
            - object arrays (one entry per trial):  z[n] -> (T_n, d_z) etc., or
            - stacked equal-length arrays:  z (N, T, d_z), x (N, T, d_x), u (N, T).
  *.pkl / *.pickle  holding a dict ``{"z_list":..., "x_list":..., "u_list":...}``
            or a 3-tuple ``(z_list, x_list, u_list)``.

Conventions (see README): per trial, ``z`` is kinematics (T_n, d_z=6), ``x`` is
non-negative integer spike counts (T_n, d_x), ``u`` is the binary perturbation
indicator (T_n,).  All three must share the same T_n within a trial.

Alternatively, just edit ``load_user_data`` below to call your own loader.

Run (use the env that has ssm + autograd installed):
    conda run -n ssm python -m perturbation_plds.run_smoke_test            # synthetic
    conda run -n ssm python -m perturbation_plds.run_smoke_test --data my.npz
"""

import argparse
import os
import pickle
import sys

import numpy as np

# Make the package importable whether run as a module or as a bare script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from perturbation_plds.fit import fit_perturbation_plds, DEFAULT_DT
from perturbation_plds.evaluate import evaluate_perturbation_plds
from perturbation_plds.crossval import crossval_compare_models
from perturbation_plds.inputs import build_augmented_inputs, build_baseline_inputs
from perturbation_plds.synthetic import generate_synthetic_data, _latent_r2


# --------------------------------------------------------------------------- #
# Data loading -- THE swap point for your own data.
# --------------------------------------------------------------------------- #
def load_user_data(path):
    """Load ``(z_list, x_list, u_list)`` from a .npz or .pkl file.

    Edit this function if your data lives in a different format/loader.
    Returns three lists (one entry per trial) of arrays shaped (T_n, d_z),
    (T_n, d_x), (T_n,).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        z, x, u = d["z"], d["x"], d["u"]
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            z, x, u = obj["z_list"], obj["x_list"], obj["u_list"]
        else:
            z, x, u = obj
        return list(z), list(x), list(u)
    else:
        raise ValueError("Unsupported data file %r (use .npz or .pkl)" % path)

    # .npz: accept either object-arrays (ragged) or stacked equal-length arrays.
    # Cast each trial to a concrete numeric dtype: an object-dtype array of equal-
    # length subarrays would otherwise propagate ``object`` dtype and break the
    # package's integer-count handling.
    z_list = [np.asarray(a, dtype=float) for a in z]
    x_list = [np.asarray(a, dtype=float) for a in x]
    u_list = [np.asarray(a, dtype=float).reshape(-1) for a in u]
    return z_list, x_list, u_list


def load_data(args):
    """Return ``(z_list, x_list, u_list, true)``; ``true`` is None for real data."""
    if args.data:
        print("Loading user data from %s ..." % args.data)
        z_list, x_list, u_list = load_user_data(args.data)
        return z_list, x_list, u_list, None

    print("No --data given; generating synthetic data (dG_true != 0) ...")
    z_list, x_list, u_list, true = generate_synthetic_data(
        p_true=args.p_true, J=args.J, N_trials=args.n_trials, T=args.T,
        d_x=args.d_x, d_z=args.d_z, dG_scale=1.0, seed=args.seed)
    return z_list, x_list, u_list, true


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _describe_data(z_list, x_list, u_list):
    Ts = [z.shape[0] for z in z_list]
    d_z = z_list[0].shape[1]
    d_x = x_list[0].shape[1]
    mc = float(np.mean([np.asarray(x).mean() for x in x_list]))
    frac_u = float(np.mean([np.asarray(u).mean() for u in u_list]))
    print("  trials=%d  T=[%d..%d]  d_z=%d  d_x=%d" %
          (len(z_list), min(Ts), max(Ts), d_z, d_x))
    print("  mean count/bin/neuron=%.3f   perturbation duty cycle=%.3f"
          % (mc, frac_u))


def smoothed_latents(results, z_list, x_list, u_list):
    """E[s_t] from the (Laplace/Kalman) smoother, for whichever model variant."""
    lds = results["_ssm_model"]
    J = results["J"]
    meta = results["_meta"]
    M = meta["M"]
    if not meta["has_G"]:
        _u, x_trim, _z, _m = build_baseline_inputs(z_list, u_list, J, x_list=x_list)
        u_aug = None
    elif meta["has_dG"]:
        u_aug, x_trim, _z, _m = build_augmented_inputs(z_list, u_list, J, x_list=x_list)
    else:
        u_aug, x_trim, _z, _m = build_baseline_inputs(z_list, u_list, J, x_list=x_list)
    _elbos, posterior = lds.approximate_posterior(
        x_trim, inputs=(u_aug if M > 0 else None), method="laplace_em", verbose=0)
    return posterior.mean_continuous_states


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="path to .npz/.pkl with your own z/x/u (default: synthetic)")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "smoke_out"),
                    help="output directory for results.pkl (default: ./smoke_out)")
    # Fit hyper-parameters.
    ap.add_argument("--p", type=int, default=6, help="latent dimension to FIT")
    ap.add_argument("--J", type=int, default=2, help="kinematic lag order")
    ap.add_argument("--lam-G", type=float, default=1e-3)
    ap.add_argument("--lam-dG", type=float, default=1e-3)
    ap.add_argument("--lam-F", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--n-iters", type=int, default=80)
    ap.add_argument("--num-init-iters", type=int, default=20)
    ap.add_argument("--verbose", type=int, default=0)
    # Cross-validation.
    ap.add_argument("--no-cv", action="store_true", help="skip held-out CV comparison")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--cv-n-iters", type=int, default=30)
    ap.add_argument("--cv-num-init-iters", type=int, default=10)
    ap.add_argument("--models", default="full,no_dG,intrinsic",
                    help="comma list of models to cross-validate")
    # Synthetic-data generation knobs (ignored when --data is given).
    ap.add_argument("--p-true", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--d-x", type=int, default=60)
    ap.add_argument("--d-z", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny/fast settings for a sanity check")
    args = ap.parse_args(argv)

    if args.quick:
        args.n_trials = min(args.n_trials, 16)
        args.T = min(args.T, 80)
        args.n_iters = min(args.n_iters, 30)
        args.num_init_iters = min(args.num_init_iters, 8)
        args.n_folds = min(args.n_folds, 2)
        args.cv_n_iters = min(args.cv_n_iters, 15)

    os.makedirs(args.out, exist_ok=True)

    print("=" * 72)
    print("PERTURBATION PLDS SMOKE TEST")
    print("=" * 72)

    z_list, x_list, u_list, true = load_data(args)
    _describe_data(z_list, x_list, u_list)

    # ---- 1. fit the full model ------------------------------------------- #
    print("\n[1] Fitting full model (p=%d, J=%d, n_iters=%d) ..."
          % (args.p, args.J, args.n_iters))
    res = fit_perturbation_plds(
        z_list, x_list, u_list, p=args.p, J=args.J, dt=args.dt,
        lam_G=args.lam_G, lam_dG=args.lam_dG, lam_F=args.lam_F,
        n_iters=args.n_iters, num_init_iters=args.num_init_iters,
        verbose=args.verbose)

    elbos = np.asarray(res["log_marginal_likelihood_trace"], float)
    diffs = np.diff(elbos)
    ascent = float(elbos[-1] - elbos[0])
    tol = max(2e-2 * abs(ascent), 1e-6 * (abs(elbos[-1]) + 1.0))
    elbo_monotone = bool(np.all(diffs > -tol)) if diffs.size else True
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(res["F"]))))
    print("  CHECK 1  ELBO non-decreasing: %s  (start=%.1f end=%.1f)"
          % (elbo_monotone, elbos[0], elbos[-1]))
    print("           spectral radius |eig(F)|_max = %.3f  (%s)"
          % (spectral_radius, "stable" if spectral_radius < 1 else "UNSTABLE"))

    # ---- 2. in-sample evaluation ----------------------------------------- #
    metrics = evaluate_perturbation_plds(res, z_list, x_list, u_list)
    print("\n[2] In-sample fit:")
    print("    deviance/obs=%.5f  pseudo-R2=%.4f  loglik=%.1f  n_obs=%d"
          % (metrics["deviance_x_per_obs"], metrics["pseudoR2_x"],
             metrics["loglik_x"], metrics["n_obs"]))

    # ---- 3. ground-truth checks (synthetic only) ------------------------- #
    eig_error = latent_r2 = None
    Es_list = smoothed_latents(res, z_list, x_list, u_list)
    if true is not None:
        ev_true = np.sort(np.abs(np.linalg.eigvals(true["F"])))[::-1]
        ev_fit = np.sort(np.abs(np.linalg.eigvals(res["F"])))[::-1]
        k = min(len(ev_true), len(ev_fit))
        eig_error = float(np.mean(np.abs(ev_true[:k] - ev_fit[:k])))
        latent_r2, _ = _latent_r2(Es_list, true["s_list"], args.J)
        print("\n[3] Ground-truth checks (synthetic):")
        print("    CHECK 2  |eig(F)| mean abs err (top-%d) = %.3f" % (k, eig_error))
        print("    CHECK 3  latent affine R^2 = %.3f  (threshold ~0.7)" % latent_r2)

    # ---- 4. cross-validated model selection ------------------------------ #
    cv = None
    if not args.no_cv:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        print("\n[4] Cross-validated model selection (n_folds=%d): %s"
              % (args.n_folds, ", ".join(models)))
        cv = crossval_compare_models(
            z_list, x_list, u_list, p=args.p, J=args.J, dt=args.dt,
            lam_G=args.lam_G, lam_dG=args.lam_dG, lam_F=args.lam_F,
            n_folds=args.n_folds, n_iters=args.cv_n_iters,
            num_init_iters=args.cv_num_init_iters, seed=args.seed, models=models)
        for m in models:
            mm = cv[m]["mean"]
            print("    %-10s held-out deviance/obs=%.5f  pseudo-R2=%.4f"
                  % (m, mm["deviance_x_per_obs"], mm["pseudoR2_x"]))
        if "full" in cv and "no_dG" in cv:
            full_wins = (cv["full"]["mean"]["deviance_x_per_obs"]
                         < cv["no_dG"]["mean"]["deviance_x_per_obs"])
            print("    => dG helps held-out fit (full < no_dG): %s" % full_wins)

    # ---- 5. save artifact ------------------------------------------------ #
    res_save = {k: v for k, v in res.items() if k != "_ssm_model"}
    artifact = dict(
        results=res_save,
        metrics=metrics,
        Es_list=[np.asarray(e) for e in Es_list],
        cv={m: dict(mean=cv[m]["mean"], std=cv[m]["std"]) for m in cv} if cv else None,
        checks=dict(elbo_monotone=elbo_monotone, spectral_radius=spectral_radius,
                    eig_error=eig_error, latent_r2=latent_r2),
        config=vars(args),
        is_synthetic=true is not None,
    )
    if true is not None:
        artifact["true"] = dict(
            F=true["F"], G_lag=true["G_lag"], dG_lag=true["dG_lag"],
            s_list=[np.asarray(s) for s in true["s_list"]],
            p=true["p"], J=true["J"])

    out_path = os.path.join(args.out, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print("\nSaved artifact -> %s" % out_path)
    print("Validate / plot with:")
    print("    conda run -n ssm python -m perturbation_plds.validate_results --in %s"
          % out_path)
    return artifact


if __name__ == "__main__":
    main()
