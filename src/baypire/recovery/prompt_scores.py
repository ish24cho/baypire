from __future__ import annotations

import numpy as np


def compute_prompt_scores(reconstructed_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(reconstructed_matrix, dtype=float).mean(axis=1)

