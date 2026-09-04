from __future__ import annotations

import pandas as pd


def ensure_binary_correctness(df: pd.DataFrame, column: str = "correctness") -> pd.DataFrame:
    out = df.copy()
    values = set(out[column].dropna().astype(float).unique().tolist())
    if not values.issubset({0.0, 1.0}):
        raise ValueError(f"{column} must be binary; found {values}")
    out[column] = out[column].astype(float)
    return out


def ensure_unique_cells(
    df: pd.DataFrame,
    keys: tuple[str, str, str] = ("model_id", "prompt_id", "item_id"),
) -> pd.DataFrame:
    duplicated = df.duplicated(list(keys), keep=False)
    if duplicated.any():
        example = df.loc[duplicated, list(keys)].head().to_dict("records")
        raise ValueError(f"Duplicate prompt-item cells found, e.g. {example}")
    return df


def sort_ids_deterministically(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return df.sort_values(list(columns), kind="mergesort").reset_index(drop=True)

