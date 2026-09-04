from __future__ import annotations

import numpy as np

from baypire.design.budget import resolve_observation_budget
from baypire.design.mask import ObservationMask


def generate_random_mask(experiment, budget: float, seed: int) -> ObservationMask:
    rng = np.random.default_rng(seed)
    I, J = experiment.shape
    n_cells = I * J
    n_observed = resolve_observation_budget(budget, n_cells)

    chosen = rng.choice(n_cells, size=n_observed, replace=False)
    observed = np.zeros(n_cells, dtype=bool)
    observed[chosen] = True
    mask = ObservationMask(observed=observed.reshape(I, J), mask_type="random_uniform", seed=seed)
    mask.validate_against(experiment)
    return mask
