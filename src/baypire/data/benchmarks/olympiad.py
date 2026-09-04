from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from baypire.data.preprocessing import ensure_binary_correctness, ensure_unique_cells, sort_ids_deterministically
from baypire.data.schema import EvaluationMatrix


def load_olympiad_long_form(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser()
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported Olympiad data path: {p}")
    return normalize_olympiad_columns(df)


def load_olympiad_matrix(path: str | Path, benchmark: str | None = None, model_id: str | None = None) -> EvaluationMatrix:
    p = Path(path).expanduser()
    if p.suffix != ".pkl":
        df = load_olympiad_long_form(p)
        if model_id is None:
            model_id = sorted(df["model_id"].astype(str).unique())[0]
        return build_olympiad_matrix(df, model_id=model_id)

    with p.open("rb") as f:
        obj = pickle.load(f)
    matrices = obj["matrices"]
    benchmark_key = benchmark or sorted(matrices.keys())[0]
    model_map = matrices[benchmark_key]
    model_key = model_id or sorted(model_map.keys())[0]
    mat = model_map[model_key]

    return EvaluationMatrix(
        Y=mat["Y"],
        prompt_ids=[str(x) for x in mat["prompt_ids"]],
        item_ids=[str(x) for x in mat["item_ids"]],
        benchmark=str(mat.get("benchmark", benchmark_key)),
        model_id=str(mat.get("model_key", model_key)),
    )


def normalize_olympiad_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    candidates = {
        "model_id": ["model_id", "model", "eval_model"],
        "prompt_id": ["prompt_id", "prompt", "prompt_name"],
        "item_id": ["item_id", "question_id", "problem_id"],
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
    out = out[["model_id", "prompt_id", "item_id", "correctness"]]
    out["model_id"] = out["model_id"].astype(str)
    out["prompt_id"] = out["prompt_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out = ensure_binary_correctness(out)
    out = ensure_unique_cells(out, keys=("model_id", "prompt_id", "item_id"))
    return sort_ids_deterministically(out, ("model_id", "prompt_id", "item_id"))


def build_olympiad_matrix(df: pd.DataFrame, model_id: str) -> EvaluationMatrix:
    model_df = df[df["model_id"].astype(str) == str(model_id)].copy()
    if model_df.empty:
        raise ValueError(f"No Olympiad rows found for model_id={model_id!r}")
    pivot = model_df.pivot(index="prompt_id", columns="item_id", values="correctness").sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Olympiad full matrix contains missing cells.")
    return EvaluationMatrix(
        Y=pivot.to_numpy(dtype=float),
        prompt_ids=pivot.index.astype(str).tolist(),
        item_ids=pivot.columns.astype(str).tolist(),
        benchmark="olympiad_math",
        model_id=str(model_id),
    )
