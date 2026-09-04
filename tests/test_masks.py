from __future__ import annotations

import numpy as np

from baypire.data.schema import EvaluationMatrix
from baypire.design.balanced_mask import generate_balanced_mask
from baypire.design.random_mask import generate_random_mask


def toy():
    return EvaluationMatrix(
        Y=np.ones((4, 5)),
        prompt_ids=[f"p{i}" for i in range(4)],
        item_ids=[f"q{j}" for j in range(5)],
        benchmark="toy",
        model_id="model",
    )


def test_balanced_mask_budget():
    mask = generate_balanced_mask(toy(), 0.2, seed=1)
    assert mask.observed.shape == (4, 5)
    assert mask.observed.sum() == 4


def test_random_mask_reproducible():
    a = generate_random_mask(toy(), 0.3, seed=7)
    b = generate_random_mask(toy(), 0.3, seed=7)
    assert np.array_equal(a.observed, b.observed)

