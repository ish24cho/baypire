from __future__ import annotations

from baypire.design.diagnostics import summarize_mask


def validate_identifiability(mask, *, require_connected: bool = False) -> dict:
    summary = summarize_mask(mask)
    if summary["prompt_coverage"] < 1.0:
        raise ValueError("At least one prompt has no observed cells.")
    if summary["item_coverage"] < 1.0:
        raise ValueError("At least one item has no observed cells.")
    if require_connected and not summary["connected"]:
        raise ValueError("The prompt-item observation graph is not connected.")
    return summary

