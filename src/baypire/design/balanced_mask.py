from __future__ import annotations

import numpy as np

from baypire.design.budget import resolve_observation_budget
from baypire.design.mask import ObservationMask


def generate_balanced_mask(experiment, budget: float, seed: int) -> ObservationMask:
    rng = np.random.default_rng(seed)
    I, J = experiment.shape
    n_cells = I * J
    n_observed = resolve_observation_budget(budget, n_cells)

    observed = np.zeros((I, J), dtype=bool)
    prompt_count = np.zeros(I, dtype=int)
    item_count = np.zeros(J, dtype=int)

    for _ in range(n_observed):
        prompt_candidates = np.flatnonzero(prompt_count == prompt_count.min())
        rng.shuffle(prompt_candidates)
        chosen = None
        for i in prompt_candidates:
            available = np.flatnonzero(~observed[i])
            if available.size:
                min_item_count = item_count[available].min()
                item_candidates = available[item_count[available] == min_item_count]
                chosen = (int(i), int(rng.choice(item_candidates)))
                break
        if chosen is None:
            break
        i, j = chosen
        observed[i, j] = True
        prompt_count[i] += 1
        item_count[j] += 1

    mask = ObservationMask(observed=observed, mask_type="two_way_balanced", seed=seed)
    mask.validate_against(experiment)
    return mask
