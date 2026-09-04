from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def prompt_mae(true_scores, estimated_scores) -> float:
    true = np.asarray(true_scores, dtype=float)
    est = np.asarray(estimated_scores, dtype=float)
    return float(np.mean(np.abs(true - est)))


def prompt_rmse(true_scores, estimated_scores) -> float:
    true = np.asarray(true_scores, dtype=float)
    est = np.asarray(estimated_scores, dtype=float)
    return float(np.sqrt(np.mean((true - est) ** 2)))


def wasserstein_1(true_scores, estimated_scores) -> float:
    return float(wasserstein_distance(np.asarray(true_scores, dtype=float), np.asarray(estimated_scores, dtype=float)))

