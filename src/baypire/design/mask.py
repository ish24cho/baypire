from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationMask:
    observed: np.ndarray
    mask_type: str
    seed: int | None = None

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=bool)
        if self.observed.ndim != 2:
            raise ValueError("Mask must be a 2D Boolean matrix.")

    @property
    def hidden(self) -> np.ndarray:
        return ~self.observed

    @property
    def observed_fraction(self) -> float:
        return float(self.observed.mean())

    @property
    def prompt_degrees(self) -> np.ndarray:
        return self.observed.sum(axis=1)

    @property
    def item_degrees(self) -> np.ndarray:
        return self.observed.sum(axis=0)

    def validate_against(self, experiment) -> None:
        if self.observed.shape != experiment.Y.shape:
            raise ValueError("Mask and evaluation matrix shapes differ.")

