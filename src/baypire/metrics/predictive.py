from __future__ import annotations

import numpy as np


def binary_log_loss(y, p, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if y.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - p) ** 2))


def hidden_log_loss(experiment, mask, mu_hat, eps: float = 1e-12) -> float:
    return binary_log_loss(experiment.Y[mask.hidden], np.asarray(mu_hat, dtype=float)[mask.hidden], eps=eps)


def hidden_brier(experiment, mask, mu_hat) -> float:
    return binary_brier(experiment.Y[mask.hidden], np.asarray(mu_hat, dtype=float)[mask.hidden])


def observed_log_loss(experiment, mask, mu_hat, eps: float = 1e-12) -> float:
    return binary_log_loss(experiment.Y[mask.observed], np.asarray(mu_hat, dtype=float)[mask.observed], eps=eps)


def observed_brier(experiment, mask, mu_hat) -> float:
    return binary_brier(experiment.Y[mask.observed], np.asarray(mu_hat, dtype=float)[mask.observed])

