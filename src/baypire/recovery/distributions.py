from __future__ import annotations

import numpy as np


def distribution_summary(prompt_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(prompt_scores, dtype=float)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(np.min(scores)),
        "q25": float(np.quantile(scores, 0.25)),
        "median": float(np.quantile(scores, 0.50)),
        "q75": float(np.quantile(scores, 0.75)),
        "max": float(np.max(scores)),
    }
