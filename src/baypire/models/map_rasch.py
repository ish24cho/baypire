from __future__ import annotations

from baypire.models._rasch_utils import fit_additive_logistic


def fit_map_rasch(
    experiment,
    mask,
    *,
    sigma_prompt: float = 1.0,
    sigma_item: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-7,
):
    prompt_penalty = 1.0 / float(sigma_prompt) ** 2
    item_penalty = 1.0 / float(sigma_item) ** 2
    mu_hat, info = fit_additive_logistic(
        experiment.Y,
        mask.observed,
        prompt_penalty=prompt_penalty,
        item_penalty=item_penalty,
        max_iter=max_iter,
        tol=tol,
    )
    info["model"] = "map_additive_rasch"
    info["sigma_prompt"] = float(sigma_prompt)
    info["sigma_item"] = float(sigma_item)
    return mu_hat, info

