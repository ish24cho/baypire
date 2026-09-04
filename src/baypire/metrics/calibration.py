from __future__ import annotations

import numpy as np


def interval_coverage(true_value, lower, upper) -> bool:
    return bool(lower <= true_value <= upper)


def interval_width(lower, upper) -> float:
    return float(upper - lower)


def prompt_interval_summary(true_scores, posterior_prompt_scores, level: float = 0.95, label: str = "") -> dict[str, float]:
    if posterior_prompt_scores is None:
        return {}
    samples = np.asarray(posterior_prompt_scores, dtype=float)
    true = np.asarray(true_scores, dtype=float)
    tail = (1.0 - float(level)) / 2.0
    lower = np.quantile(samples, tail, axis=0)
    upper = np.quantile(samples, 1.0 - tail, axis=0)
    width = upper - lower
    covered = (true >= lower) & (true <= upper)
    suffix = int(round(level * 100))
    prefix = f"{label}_" if label else ""
    return {
        f"cover_prompt_{prefix}{suffix}": float(np.mean(covered)),
        f"mean_interval_width_{prefix}{suffix}": float(np.mean(width)),
        f"median_interval_width_{prefix}{suffix}": float(np.median(width)),
    }
