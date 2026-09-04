from __future__ import annotations

from pathlib import Path

import pandas as pd

from baypire.data.preprocessing import ensure_binary_correctness, ensure_unique_cells, sort_ids_deterministically
from baypire.data.schema import EvaluationMatrix


def load_mmlu_long_form(path: str | Path) -> pd.DataFrame:
    """Load normalized PromptEval/MMLU correctness data.

    The expected input is either a directory of subject parquet chunks or one
    parquet/CSV file. The adapter normalizes column names to:
    model_id, prompt_id, item_id, subject, correctness.
    """
    p = Path(path).expanduser()
    pieces = []
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet chunks found in {p}")
        for file in files:
            pieces.append(pd.read_parquet(file))
    elif p.suffix == ".parquet":
        pieces.append(pd.read_parquet(p))
    elif p.suffix == ".csv":
        pieces.append(pd.read_csv(p))
    else:
        raise ValueError(f"Unsupported MMLU data path: {p}")

    df = pd.concat(pieces, ignore_index=True)
    df = normalize_mmlu_columns(df)
    df = ensure_binary_correctness(df)
    df = ensure_unique_cells(df, keys=("model_id", "prompt_id", "item_id"))
    return sort_ids_deterministically(df, ("model_id", "subject", "prompt_id", "item_id"))


def normalize_mmlu_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    candidates = {
        "model_id": ["model_id", "model", "eval_model", "model_key"],
        "prompt_id": ["prompt_id", "prompt", "format", "prompt_name"],
        "item_id": ["item_id", "question_id", "example_id", "id"],
        "subject": ["subject", "task", "category"],
        "correctness": ["correctness", "correct", "score", "is_correct"],
    }
    for canonical, options in candidates.items():
        for option in options:
            if option in df.columns:
                rename[option] = canonical
                break
        if canonical not in rename.values() and canonical not in df.columns:
            raise ValueError(f"Could not find a column for {canonical}; available columns={list(df.columns)}")

    out = df.rename(columns=rename).copy()
    out = out[["model_id", "prompt_id", "item_id", "subject", "correctness"]]
    out["model_id"] = out["model_id"].astype(str)
    out["prompt_id"] = out["prompt_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["subject"] = out["subject"].astype(str)
    return out


def build_mmlu_matrix(df: pd.DataFrame, model_id: str, subject: str | None = None) -> EvaluationMatrix:
    model_df = df[df["model_id"].astype(str) == str(model_id)].copy()
    if subject is not None:
        model_df = model_df[model_df["subject"].astype(str) == str(subject)].copy()
    if model_df.empty:
        raise ValueError(f"No MMLU rows found for model_id={model_id!r}, subject={subject!r}")

    pivot = model_df.pivot(index="prompt_id", columns="item_id", values="correctness").sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Selected MMLU matrix is not complete.")

    benchmark = "mmlu" if subject is None else f"mmlu:{subject}"
    item_meta = model_df[["item_id", "subject"]].drop_duplicates("item_id").set_index("item_id").loc[pivot.columns]
    return EvaluationMatrix(
        Y=pivot.to_numpy(dtype=float),
        prompt_ids=pivot.index.astype(str).tolist(),
        item_ids=pivot.columns.astype(str).tolist(),
        benchmark=benchmark,
        model_id=str(model_id),
        item_metadata=item_meta.reset_index(),
    )

