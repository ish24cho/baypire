from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np

from baypire.models._rasch_utils import fit_additive_logistic


def fit_pe_rasch(
    experiment,
    mask,
    *,
    prompteval_repo: str | Path | None = None,
    ridge: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-7,
):
    """Fit PromptEval-style Rasch and return full-matrix probabilities.

    If `prompteval_repo` points to a local official PromptEval clone, this uses
    `prompteval.methods.ExtendedRaschModel`. Otherwise it uses the local
    additive logistic implementation with a very weak ridge penalty.
    """
    official = load_official_methods(prompteval_repo) if prompteval_repo else None
    if official is not None:
        model = official.ExtendedRaschModel()
        model.fit(mask.observed.copy(), experiment.Y)
        mu_hat = np.asarray(official.sigmoid(model.logits), dtype=float)
        return np.clip(mu_hat, 1e-8, 1.0 - 1e-8), {
            "converged": True,
            "model": "official_prompteval.ExtendedRaschModel",
            "n_obs": int(mask.observed.sum()),
        }

    mu_hat, info = fit_additive_logistic(
        experiment.Y,
        mask.observed,
        prompt_penalty=float(ridge),
        item_penalty=float(ridge),
        max_iter=max_iter,
        tol=tol,
    )
    info["model"] = "local_weakly_regularized_rasch"
    info["ridge"] = float(ridge)
    return mu_hat, info


def load_official_methods(repo_path: str | Path | None):
    if repo_path is None:
        return None
    repo = Path(repo_path).expanduser().resolve()
    if not (repo / "prompteval/methods.py").exists():
        return None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # The official methods module imports prompteval.utils, whose full module
    # imports torch/transformers. ExtendedRaschModel only needs this symbol.
    if "prompteval.utils" not in sys.modules:
        utils_stub = types.ModuleType("prompteval.utils")

        def check_multicolinearity(_X, _tol=1e-6):
            return None

        utils_stub.check_multicolinearity = check_multicolinearity
        sys.modules["prompteval.utils"] = utils_stub
    return importlib.import_module("prompteval.methods")

