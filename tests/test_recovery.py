from __future__ import annotations

import numpy as np

from baypire.data.schema import EvaluationMatrix
from baypire.design.mask import ObservationMask
from baypire.recovery.imputation import reconstruct_matrix


def test_observed_retention():
    Y = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    experiment = EvaluationMatrix(Y, ["p0", "p1"], ["q0", "q1", "q2"], "toy", "model")
    mask = ObservationMask(
        np.array([[True, False, True], [False, True, False]]),
        "toy",
    )
    mu_hat = np.full_like(Y, 0.5)
    reconstructed = reconstruct_matrix(experiment, mask, mu_hat)
    assert np.array_equal(reconstructed[mask.observed], Y[mask.observed])
    assert np.all(reconstructed[mask.hidden] == 0.5)

