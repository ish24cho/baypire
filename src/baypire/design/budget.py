from __future__ import annotations


def resolve_observation_budget(budget: float, n_cells: int) -> int:
    """Interpret budget as a fraction when <=1, otherwise as an absolute count."""
    value = float(budget)
    if value <= 0:
        raise ValueError("Observation budget must be positive.")
    if n_cells <= 0:
        raise ValueError("Number of cells must be positive.")

    n_observed = int(round(value * n_cells)) if value <= 1 else int(round(value))
    return min(max(n_observed, 1), n_cells)
