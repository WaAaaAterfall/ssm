"""Fit the full perturbation PLDS at p=10 on session 0 and cache everything the
manifold/dimensionality analysis needs (params + inferred per-trial latents +
trimmed kinematics/indicators).  Run once (~15 min); the notebook loads the pkl.
"""

import os
import pickle
import time

import numpy as np

from perturbation_plds import fit_perturbation_plds, infer_latents

SESSION = 0
WINDOW = 168
P = 10
J = 3
LAM = 1e-3
N_ITERS = 60
NUM_INIT_ITERS = 12
OUT = "perturbation_plds_results/session0/fit_p10.pkl"

DATA = (f"/home/sp645/isilon/All_Staff/sp645/data/Yi_deconvoled_calcium/"
        f"Ephys_no_smooth/session{SESSION}.pkl")


DATA_TMPL = ("/home/sp645/isilon/All_Staff/sp645/data/Yi_deconvoled_calcium/"
             "Ephys_no_smooth/session{n}.pkl")


def load_session(session=SESSION, window=WINDOW):
    with open(DATA_TMPL.format(n=session), "rb") as f:
        obj = pickle.load(f)
    behavior = obj["behavior"][:, :window, :]
    train = obj["train"][:, :window, :]
    Y = behavior[:, :, :2]
    vel = Y[:, 1:, :] - Y[:, :-1, :]
    vel = np.concatenate([vel, np.zeros((train.shape[0], 1, 2))], axis=1)
    Z = np.concatenate([Y, vel], axis=-1)      # (N, T, 4)
    U = behavior[:, :, 2]                       # (N, T)
    X = train                                  # (N, T, d_x)
    return Z, U, X


def fit_and_cache(session, out_path, *, window=WINDOW, p=P, J=J, lam=LAM,
                  n_iters=N_ITERS, num_init_iters=NUM_INIT_ITERS):
    """Fit the full perturbation PLDS for one session and cache everything the
    manifold/dimensionality + drive-angle analyses need.  Returns the cache dict."""
    t0 = time.time()
    Z, U, X = load_session(session, window)
    print(f"[s{session}] data: Z{Z.shape} U{U.shape} X{X.shape}", flush=True)

    res = fit_perturbation_plds(Z, X, U, p=p, J=J, lam_G=lam, lam_dG=lam,
                                n_iters=n_iters, num_init_iters=num_init_iters,
                                verbose=2)
    print(f"[s{session}] fit done in {(time.time()-t0)/60:.1f} min; "
          f"final ELBO {res['log_marginal_likelihood_trace'][-1]:.1f}", flush=True)

    latents = infer_latents(res, Z, X, U, post_n_iters=15)
    print(f"[s{session}] inferred {len(latents)} latents, e.g. {latents[0].shape}",
          flush=True)

    # per-trial trimmed indicator/kinematics aligned to the latents (x[J:]);
    # keep the *untrimmed* arrays too for the counterfactual rollout.
    u_trim = [np.asarray(U[n])[J:] for n in range(U.shape[0])]
    z_trim = [np.asarray(Z[n])[J:] for n in range(Z.shape[0])]

    cache = dict(
        F=res["F"], Q=res["Q"], m_s=res["m_s"],
        G_lag=res["G_lag"], dG_lag=res["dG_lag"],
        C=res["C"], b=res["b"], p=res["p"], J=res["J"], dt=res["dt"],
        meta=res["_meta"],
        elbo_trace=res["log_marginal_likelihood_trace"],
        latents=latents,            # list of (T-J, p)
        u_trim=u_trim, z_trim=z_trim,
        Z_full=Z, U_full=U,         # untrimmed, for counterfactual
        config=dict(session=session, window=window, p=p, J=J, lam=lam,
                    n_iters=n_iters, num_init_iters=num_init_iters),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[s{session}] wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB) "
          f"total {(time.time()-t0)/60:.1f} min", flush=True)
    return cache


def main():
    fit_and_cache(SESSION, OUT)


if __name__ == "__main__":
    main()
