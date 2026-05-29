"""Trial-level cross-validation for the perturbation PLDS (Task 6).

Folds are split by *trial index* only -- never by within-trial bin -- so train
and test sets contain disjoint trials.  The same routine drives the full model
and both nested ablations, so the user can compare held-out ``deviance_x``
across models (the actual scientific test of whether dG is real).
"""

import numpy as np

from .fit import (fit_perturbation_plds,
                  fit_perturbation_plds_no_dG,
                  fit_perturbation_plds_intrinsic)
from .evaluate import evaluate_perturbation_plds

_FITTERS = dict(
    full=fit_perturbation_plds,
    no_dG=fit_perturbation_plds_no_dG,
    intrinsic=fit_perturbation_plds_intrinsic,
)


def _kfold_trial_indices(n_trials, n_folds, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_trials)
    return [perm[i::n_folds] for i in range(n_folds)]


def crossval_perturbation_plds(z_list, x_list, u_list, *, p, J, n_folds=5,
                               lam_G=1e-4, lam_dG=1e-4, lam_F=0.0,
                               model="full", dt=5.0 / 240.0, n_iters=100,
                               seed=0, verbose=0, num_init_iters=25):
    """Cross-validate one model variant.

    Parameters
    ----------
    model : {'full', 'no_dG', 'intrinsic'}
        Which fitter to cross-validate.
    n_folds : int
        Number of trial-level folds.

    Returns
    -------
    dict with ``per_fold`` (list of metric dicts), and ``mean``/``std`` dicts of
    the scalar metrics across folds, plus ``model``, ``p``, ``J``.
    """
    if model not in _FITTERS:
        raise ValueError("model must be one of %s" % list(_FITTERS))
    fitter = _FITTERS[model]

    n_trials = len(x_list)
    assert n_trials >= n_folds, "need at least n_folds trials"
    folds = _kfold_trial_indices(n_trials, n_folds, seed=seed)

    fit_kwargs = dict(p=p, J=J, dt=dt, n_iters=n_iters, verbose=verbose,
                      num_init_iters=num_init_iters, lam_F=lam_F)
    if model != "intrinsic":
        fit_kwargs.update(lam_G=lam_G, lam_dG=lam_dG)

    per_fold = []
    for k, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        train_idx = [i for i in range(n_trials) if i not in test_set]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        z_tr = [z_list[i] for i in train_idx]
        x_tr = [x_list[i] for i in train_idx]
        u_tr = [u_list[i] for i in train_idx]
        z_te = [z_list[i] for i in test_idx]
        x_te = [x_list[i] for i in test_idx]
        u_te = [u_list[i] for i in test_idx]

        results = fitter(z_tr, x_tr, u_tr, **fit_kwargs)
        metrics = evaluate_perturbation_plds(results, z_te, x_te, u_te)
        metrics["fold"] = k
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        per_fold.append(metrics)

    scalar_keys = [k for k in per_fold[0]
                   if isinstance(per_fold[0][k], (int, float))
                   and k not in ("fold", "n_train", "n_test", "n_obs")]
    mean = {k: float(np.mean([m[k] for m in per_fold])) for k in scalar_keys}
    std = {k: float(np.std([m[k] for m in per_fold])) for k in scalar_keys}

    return dict(model=model, p=p, J=J, per_fold=per_fold, mean=mean, std=std)


def crossval_compare_models(z_list, x_list, u_list, *, p, J, n_folds=5,
                            lam_G=1e-4, lam_dG=1e-4, lam_F=0.0,
                            dt=5.0 / 240.0, n_iters=100, seed=0, verbose=0,
                            num_init_iters=25, models=("full", "no_dG", "intrinsic")):
    """Run :func:`crossval_perturbation_plds` for several model variants.

    Returns a dict mapping model name -> crossval result dict.  Compare
    ``out[m]['mean']['deviance_x']`` across models; lower is better.
    """
    out = {}
    for model in models:
        out[model] = crossval_perturbation_plds(
            z_list, x_list, u_list, p=p, J=J, n_folds=n_folds,
            lam_G=lam_G, lam_dG=lam_dG, lam_F=lam_F, model=model,
            dt=dt, n_iters=n_iters, seed=seed, verbose=verbose,
            num_init_iters=num_init_iters)
    return out
