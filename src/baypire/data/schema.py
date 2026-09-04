from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationMatrix:
    Y: np.ndarray
    prompt_ids: list[str]
    item_ids: list[str]
    benchmark: str
    model_id: str
    prompt_metadata: pd.DataFrame | None = None
    item_metadata: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.Y = np.asarray(self.Y, dtype=float)

        if self.Y.ndim != 2:
            raise ValueError("Y must be a 2D matrix.")
        if self.Y.shape[0] != len(self.prompt_ids):
            raise ValueError("Number of prompts does not match Y.")
        if self.Y.shape[1] != len(self.item_ids):
            raise ValueError("Number of items does not match Y.")

        finite = self.Y[np.isfinite(self.Y)]
        if finite.size == 0:
            raise ValueError("Y must contain at least one finite cell.")
        if not set(np.unique(finite).tolist()).issubset({0.0, 1.0}):
            raise ValueError("Y must contain only binary correctness values.")
        if not np.all(np.isfinite(self.Y)):
            raise ValueError("EvaluationMatrix must be complete after benchmark filtering.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.Y.shape

    @property
    def n_prompts(self) -> int:
        return self.Y.shape[0]

    @property
    def n_items(self) -> int:
        return self.Y.shape[1]

    @property
    def full_prompt_scores(self) -> np.ndarray:
        return self.Y.mean(axis=1)

