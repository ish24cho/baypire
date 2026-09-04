from __future__ import annotations

import numpy as np


def reconstruct_matrix(experiment, mask, predicted_probabilities: np.ndarray) -> np.ndarray:
    mu_hat = np.asarray(predicted_probabilities, dtype=float)
    if mu_hat.shape != experiment.Y.shape:
        raise ValueError("Predicted probability matrix has wrong shape.")

    reconstructed = mu_hat.copy()
    reconstructed[mask.observed] = experiment.Y[mask.observed]
    return reconstructed

