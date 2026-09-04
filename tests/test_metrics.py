from __future__ import annotations

import numpy as np

from baypire.metrics.recovery import prompt_mae, prompt_rmse, wasserstein_1


def test_recovery_metrics():
    true = np.array([0.0, 1.0])
    est = np.array([0.5, 0.5])
    assert prompt_mae(true, est) == 0.5
    assert prompt_rmse(true, est) == 0.5
    assert wasserstein_1(true, est) == 0.5
