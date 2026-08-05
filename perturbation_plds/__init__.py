"""Customized Poisson Linear Dynamical System (PLDS) with perturbation-gated
kinematic inputs, built on top of the ``ssm`` library.

See README.md for the model specification and the preprocessing trick.
"""

from .inputs import build_augmented_inputs, build_baseline_inputs
from .model import PerBlockRidgeGaussianDynamics, NoInputPoissonEmissions
from .fit import (fit_perturbation_plds,
                  fit_perturbation_plds_no_dG,
                  fit_perturbation_plds_intrinsic)
from .evaluate import evaluate_perturbation_plds, infer_latents
from .crossval import (crossval_perturbation_plds,
                       crossval_compare_models)
from .synthetic import generate_synthetic_data, run_smoke_test
from . import manifold_analysis

__all__ = [
    "build_augmented_inputs", "build_baseline_inputs",
    "PerBlockRidgeGaussianDynamics", "NoInputPoissonEmissions",
    "fit_perturbation_plds", "fit_perturbation_plds_no_dG",
    "fit_perturbation_plds_intrinsic",
    "evaluate_perturbation_plds", "infer_latents",
    "crossval_perturbation_plds", "crossval_compare_models",
    "generate_synthetic_data", "run_smoke_test",
    "manifold_analysis",
]
