from __future__ import annotations

from collections import deque

import numpy as np


def prompt_coverage(mask) -> float:
    return float(np.mean(mask.prompt_degrees > 0))


def item_coverage(mask) -> float:
    return float(np.mean(mask.item_degrees > 0))


def is_connected(mask) -> bool:
    observed = np.asarray(mask.observed, dtype=bool)
    I, J = observed.shape
    if I == 0 or J == 0:
        return False

    total_nodes = I + J
    adj = [[] for _ in range(total_nodes)]
    rows, cols = np.where(observed)
    for i, j in zip(rows, cols):
        p_node = int(i)
        q_node = I + int(j)
        adj[p_node].append(q_node)
        adj[q_node].append(p_node)

    if any(len(neighbours) == 0 for neighbours in adj):
        return False

    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == total_nodes


def summarize_mask(mask) -> dict[str, float | int | bool]:
    return {
        "observed_fraction": mask.observed_fraction,
        "prompt_coverage": prompt_coverage(mask),
        "item_coverage": item_coverage(mask),
        "connected": is_connected(mask),
        "min_prompt_degree": int(mask.prompt_degrees.min()),
        "max_prompt_degree": int(mask.prompt_degrees.max()),
        "min_item_degree": int(mask.item_degrees.min()),
        "max_item_degree": int(mask.item_degrees.max()),
    }

