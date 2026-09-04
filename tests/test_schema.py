from __future__ import annotations

import numpy as np
import pytest

from baypire.data.schema import EvaluationMatrix


def test_evaluation_matrix_shape():
    experiment = EvaluationMatrix(
        Y=np.array([[1, 0], [0, 1]], dtype=float),
        prompt_ids=["p0", "p1"],
        item_ids=["q0", "q1"],
        benchmark="toy",
        model_id="model",
    )
    assert experiment.shape == (2, 2)
    assert experiment.full_prompt_scores.tolist() == [0.5, 0.5]


def test_reject_nonbinary_y():
    with pytest.raises(ValueError):
        EvaluationMatrix(
            Y=np.array([[0.2, 1.0]]),
            prompt_ids=["p0"],
            item_ids=["q0", "q1"],
            benchmark="toy",
            model_id="model",
        )

