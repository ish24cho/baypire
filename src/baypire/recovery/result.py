from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RecoveryResult:
    method: str
    predicted_probabilities: np.ndarray
    reconstructed_matrix: np.ndarray
    prompt_scores: np.ndarray
    posterior_prompt_scores: np.ndarray | None = None
    posterior_predictive_prompt_scores: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicted_probabilities = np.asarray(self.predicted_probabilities, dtype=float)
        self.reconstructed_matrix = np.asarray(self.reconstructed_matrix, dtype=float)
        self.prompt_scores = np.asarray(self.prompt_scores, dtype=float)

        if self.reconstructed_matrix.shape != self.predicted_probabilities.shape:
            raise ValueError("Recovery matrices must have the same shape.")
        if self.prompt_scores.shape != (self.reconstructed_matrix.shape[0],):
            raise ValueError("Need exactly one recovered score per prompt.")
