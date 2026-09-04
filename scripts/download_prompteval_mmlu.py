#!/usr/bin/env python3
"""Download/import released PromptEval MMLU correctness chunks.

Usage:
    python -m baypire.scripts.download_prompteval_mmlu --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DATASET_NAME = "PromptEval/PromptEval_MMLU_correctness"
STANDARD_COLUMNS = [
    "model",
    "prompt",
    "prompt_index",
    "question_id",
    "subject",
    "correctness",
    "source_dataset",
    "source_config",
    "source_split",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import PromptEval MMLU correctness from Hugging Face.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--chunks-dir", default="data/external/prompteval_mmlu/chunks")
    parser.add_argument("--combined-out", default="data/processed/mmlu/correctness.parquet")
    parser.add_argument("--subjects", nargs="*", help="Optional subject/config names. Default: all remote subjects.")
    parser.add_argument("--models", nargs="*", default=["meta_llama_llama_3_8b"], help="Model split names, or 'all'.")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--force", action="store_true", help="Rebuild existing chunks.")
    parser.add_argument("--combine", action="store_true", help="Also write one combined processed parquet.")
    parser.add_argument("--list-remote", action="store_true", help="Only list remote subjects and model splits.")
    args = parser.parse_args()

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    subjects = select_names(
        sorted(get_dataset_config_names(args.dataset)),
        args.subjects,
        args.max_subjects,
    )

    if args.list_remote:
        print(f"dataset={args.dataset}")
        print(f"n_subjects={len(subjects)}")
        print("\n".join(subjects))
        example_subject = subjects[0]
        splits = sorted(get_dataset_split_names(args.dataset, example_subject))
        print(f"\nexample_subject={example_subject}")
        print(f"n_model_splits={len(splits)}")
        print("\n".join(splits[:50]))
        return

    chunks_dir = resolve(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        chunk_path = chunks_dir / f"{subject}.parquet"
        if chunk_path.exists() and not args.force:
            print(f"Skip existing subject chunk: {subject}")
            continue

        remote_splits = sorted(get_dataset_split_names(args.dataset, subject))
        if args.models == ["all"]:
            models = remote_splits
        else:
            models = select_names(remote_splits, args.models, args.max_models)

        frames = []
        for model in models:
            print(f"Loading subject={subject} model={model}")
            ds = load_dataset(args.dataset, subject, split=model)
            frame = standardise_correctness_frame(
                ds.to_pandas(),
                dataset=args.dataset,
                subject=subject,
                model=model,
                max_prompts=args.max_prompts,
                max_questions=args.max_questions,
            )
            frames.append(frame)

        subject_df = optimise_types(pd.concat(frames, ignore_index=True))
        subject_df.to_parquet(chunk_path, index=False)
        print(
            f"Wrote {chunk_path} "
            f"({len(subject_df)} rows, {subject_df['model'].nunique()} models, "
            f"{subject_df['prompt'].nunique()} prompts, {subject_df['question_id'].nunique()} questions)"
        )

    manifest = write_manifest(args.dataset, chunks_dir)
    print(f"Subject chunks ready: {manifest['n_subject_chunks']}")

    if args.combine:
        combined_out = resolve(args.combined_out)
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        pieces = [pd.read_parquet(path) for path in sorted(chunks_dir.glob("*.parquet"))]
        df = optimise_types(pd.concat(pieces, ignore_index=True))
        df.to_parquet(combined_out, index=False)
        print(f"Combined parquet: {combined_out} ({len(df)} rows)")


def standardise_correctness_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    subject: str,
    model: str,
    max_prompts: int | None,
    max_questions: int | None,
) -> pd.DataFrame:
    example_cols = [c for c in frame.columns if str(c).startswith("example_")]
    example_cols = sorted(example_cols, key=example_sort_key)
    if max_questions is not None:
        example_cols = example_cols[: int(max_questions)]
    if not example_cols:
        raise ValueError(f"No example_* columns found for subject={subject}, model={model}")

    prompt_frame = frame.reset_index(names="prompt_index")
    prompt_frame["prompt"] = prompt_frame["prompt_index"].map(lambda i: f"format_{int(i)}")
    if max_prompts is not None:
        prompt_frame = prompt_frame.iloc[: int(max_prompts)].copy()

    long = prompt_frame.melt(
        id_vars=["prompt_index", "prompt"],
        value_vars=example_cols,
        var_name="example",
        value_name="correctness",
    )
    long["model"] = model
    long["subject"] = subject
    long["question_id"] = long["subject"] + ":" + long["example"].astype(str)
    long["source_dataset"] = dataset
    long["source_config"] = subject
    long["source_split"] = model
    long = long[STANDARD_COLUMNS]
    long["correctness"] = pd.to_numeric(long["correctness"], errors="coerce")
    long = long.dropna(subset=["correctness"])
    values = set(long["correctness"].astype(float).unique().tolist())
    if not values.issubset({0.0, 1.0}):
        raise ValueError(f"correctness must be binary for subject={subject}, model={model}; found {values}")
    return long


def optimise_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["correctness"] = out["correctness"].astype("int8")
    for col in ["model", "prompt", "subject", "source_dataset", "source_config", "source_split"]:
        out[col] = out[col].astype("category")
    return out


def write_manifest(dataset: str, chunks_dir: Path) -> dict:
    rows = []
    total = 0
    for path in sorted(chunks_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["model", "prompt", "question_id", "correctness"])
        n = len(df)
        total += n
        rows.append(
            {
                "subject": path.stem,
                "path": str(path),
                "n_observations": int(n),
                "n_models": int(df["model"].nunique()),
                "n_prompts": int(df["prompt"].nunique()),
                "n_questions": int(df["question_id"].nunique()),
            }
        )
    manifest = {
        "dataset": dataset,
        "chunks_dir": str(chunks_dir),
        "n_subject_chunks": len(rows),
        "n_observations": int(total),
        "subjects": rows,
    }
    out = chunks_dir.parent / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def select_names(discovered: list[str], requested: Iterable[str] | None, limit: int | None) -> list[str]:
    if requested:
        names = [str(x) for x in requested]
        missing = sorted(set(names) - set(discovered))
        if missing:
            raise ValueError(f"Requested names not found: {missing}")
    else:
        names = discovered
    if limit is not None:
        names = names[: int(limit)]
    return names


def example_sort_key(name: str) -> tuple[int, str]:
    suffix = str(name).split("example_", 1)[-1]
    try:
        return (int(suffix), str(name))
    except ValueError:
        return (10**9, str(name))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


if __name__ == "__main__":
    main()
