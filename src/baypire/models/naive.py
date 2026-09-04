from __future__ import annotations

import numpy as np


def predict_naive(experiment, mask) -> np.ndarray:
    Y = experiment.Y
    mu_hat = np.zeros_like(Y, dtype=float)
    global_mean = float(Y[mask.observed].mean()) if mask.observed.any() else float(Y.mean())

    for i in range(experiment.n_prompts):
        obs = mask.observed[i]
        row_mean = float(Y[i, obs].mean()) if obs.any() else global_mean
        mu_hat[i, :] = row_mean

    return np.clip(mu_hat, 1e-8, 1.0 - 1e-8)

