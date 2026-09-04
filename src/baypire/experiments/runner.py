from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from baypire.design.balanced_mask import generate_balanced_mask
from baypire.design.diagnostics import summarize_mask
from baypire.design.random_mask import generate_random_mask
from baypire.metrics.calibration import prompt_interval_summary
from baypire.metrics.predictive import hidden_brier, hidden_log_loss, observed_brier, observed_log_loss
from baypire.metrics.recovery import prompt_mae, prompt_rmse, wasserstein_1
from baypire.recovery.imputation import reconstruct_matrix
from baypire.recovery.prompt_scores import compute_prompt_scores
from baypire.recovery.result import RecoveryResult


@dataclass
class MethodSpec:
    name: str
    fit: Callable
    kwargs: dict[str, Any] | None = None


def build_recovery_result(
    method,
    experiment,
    mask,
    mu_hat,
    posterior_prompt_scores=None,
    posterior_predictive_prompt_scores=None,
    diagnostics=None,
):
    reconstructed = reconstruct_matrix(experiment, mask, mu_hat)
    scores = compute_prompt_scores(reconstructed)
    return RecoveryResult(
        method=method,
        predicted_probabilities=mu_hat,
        reconstructed_matrix=reconstructed,
        prompt_scores=scores,
        posterior_prompt_scores=posterior_prompt_scores,
        posterior_predictive_prompt_scores=posterior_predictive_prompt_scores,
        diagnostics=diagnostics or {},
    )


def run_experiment(
    experiment,
    methods: list[MethodSpec],
    budgets=(200, 400, 800, 1600),
    n_masks: int = 20,
    mask_type: str = "balanced",
    base_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, RecoveryResult]]:
    rows = []
    latest_results: dict[str, RecoveryResult] = {}
    true_scores = experiment.full_prompt_scores

    for budget in budgets:
        for repeat in range(n_masks):
            seed = base_seed + 1000 * repeat + int(round(float(budget) * 1000))
            mask = _make_mask(experiment, budget, seed, mask_type)
            mask_summary = summarize_mask(mask)

            for spec in methods:
                prediction = spec.fit(experiment, mask, **(spec.kwargs or {}))
                mu_hat = getattr(prediction, "mean_probabilities", prediction)
                diagnostics = getattr(prediction, "diagnostics", {})
                posterior_prompt_scores = getattr(prediction, "posterior_prompt_scores", None)
                posterior_predictive_prompt_scores = getattr(prediction, "posterior_predictive_prompt_scores", None)
                if isinstance(prediction, tuple):
                    mu_hat, diagnostics = prediction
                    posterior_prompt_scores = None
                    posterior_predictive_prompt_scores = None

                result = build_recovery_result(
                    spec.name,
                    experiment,
                    mask,
                    mu_hat,
                    posterior_prompt_scores=posterior_prompt_scores,
                    posterior_predictive_prompt_scores=posterior_predictive_prompt_scores,
                    diagnostics=diagnostics,
                )

                row = {
                    "benchmark": experiment.benchmark,
                    "model_id": experiment.model_id,
                    "budget": float(budget),
                    "repeat": int(repeat),
                    "mask_type": mask.mask_type,
                    "method": spec.name,
                    "n_prompts": experiment.n_prompts,
                    "n_items": experiment.n_items,
                    "n_observed": int(mask.observed.sum()),
                    "n_total": int(mask.observed.size),
                    "prompt_mae": prompt_mae(true_scores, result.prompt_scores),
                    "prompt_rmse": prompt_rmse(true_scores, result.prompt_scores),
                    "wasserstein_1": wasserstein_1(true_scores, result.prompt_scores),
                    "test_log_loss": hidden_log_loss(experiment, mask, mu_hat),
                    "test_brier": hidden_brier(experiment, mask, mu_hat),
                    "train_log_loss": observed_log_loss(experiment, mask, mu_hat),
                    "train_brier": observed_brier(experiment, mask, mu_hat),
                    **{f"mask_{k}": v for k, v in mask_summary.items()},
                    **diagnostics,
                    **prompt_interval_summary(true_scores, posterior_prompt_scores, label="latent"),
                    **prompt_interval_summary(true_scores, posterior_predictive_prompt_scores, label="predictive"),
                }
                if posterior_prompt_scores is not None:
                    row.update(_average_interval_summary(true_scores, posterior_prompt_scores, label="latent"))
                if posterior_predictive_prompt_scores is not None:
                    row.update(
                        _average_interval_summary(
                            true_scores,
                            posterior_predictive_prompt_scores,
                            label="predictive",
                        )
                    )

                rows.append(row)
                latest_results[f"{spec.name}:b{budget}:r{repeat}"] = result

    return pd.DataFrame(rows), latest_results


def _make_mask(experiment, budget: float, seed: int, mask_type: str):
    if mask_type == "balanced":
        return generate_balanced_mask(experiment, budget, seed)
    if mask_type == "random":
        return generate_random_mask(experiment, budget, seed)
    raise ValueError(f"Unknown mask_type={mask_type!r}")


def _average_interval_summary(true_scores, posterior_prompt_scores, level: float = 0.95, label: str = ""):
    samples = np.asarray(posterior_prompt_scores, dtype=float).mean(axis=1)
    tail = (1.0 - float(level)) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    true_avg = float(np.mean(true_scores))
    suffix = int(round(level * 100))
    prefix = f"{label}_" if label else ""
    return {
        f"cover_avg_{prefix}{suffix}": float(lower <= true_avg <= upper),
        f"avg_interval_width_{prefix}{suffix}": float(upper - lower),
    }
