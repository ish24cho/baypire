#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baypire.data.benchmarks.mmlu import load_mmlu_long_form


DEFAULT_DONE = {"abstract_algebra", "anatomy"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU recovery experiments over multiple subjects.")
    parser.add_argument("--data", default="data/external/prompteval_mmlu/chunks")
    parser.add_argument("--model-id", default="meta_llama_llama_3_8b")
    parser.add_argument("--subjects", nargs="+", default=["remaining"], help="'all', 'remaining', or explicit subjects.")
    parser.add_argument("--mask-types", nargs="+", default=["balanced", "random"], choices=["balanced", "random"])
    parser.add_argument("--budgets", default="200,400,800,1600")
    parser.add_argument("--n-masks", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=["Avg", "PE-Rasch", "Bayes-MAP", "BayPIRE"])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--results-dir", default="results/mmlu")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = resolve_subjects(args)
    jobs = [(subject, mask_type) for subject in subjects for mask_type in args.mask_types]
    print(f"model_id={args.model_id}")
    print(f"subjects={len(subjects)}")
    print(f"jobs={len(jobs)}")

    if args.dry_run:
        for subject, mask_type in jobs:
            print(f"{subject} {mask_type} -> {summary_path(args.results_dir, args.model_id, subject, mask_type)}")
        return

    for subject, mask_type in jobs:
        out_dir = resolve(args.results_dir) / safe_path_component(args.model_id) / subject / mask_type
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = summary_path(args.results_dir, args.model_id, subject, mask_type)
        if args.skip_existing and summary.exists():
            print(f"skip existing {subject} {mask_type}: {summary}")
            continue

        log_path = out_dir / "run.log"
        result_path = out_dir / f"{mask_type}.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_mmlu.py"),
            "--data",
            args.data,
            "--model-id",
            args.model_id,
            "--subject",
            subject,
            "--budgets",
            args.budgets,
            "--n-masks",
            str(args.n_masks),
            "--mask-type",
            mask_type,
            "--methods",
            *args.methods,
            "--draws",
            str(args.draws),
            "--tune",
            str(args.tune),
            "--chains",
            str(args.chains),
            "--target-accept",
            str(args.target_accept),
            "--out",
            str(result_path),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        print(f"run {subject} {mask_type}")
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"done {subject} {mask_type}: {summary}")
        write_balanced_vs_random_summary(args.results_dir, args.model_id, subject)


def resolve_subjects(args) -> list[str]:
    requested = args.subjects
    df = load_mmlu_long_form(resolve(args.data))
    available = sorted(df["subject"].astype(str).unique())
    if requested == ["all"]:
        return available
    if requested == ["remaining"]:
        return [subject for subject in available if subject not in DEFAULT_DONE]

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown subjects: {missing}")
    return requested


def write_balanced_vs_random_summary(results_dir: str, model_id: str, subject: str) -> None:
    base = resolve(results_dir) / safe_path_component(model_id) / subject
    frames = []
    for mask_type in ["balanced", "random"]:
        path = base / mask_type / f"{mask_type}_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "experiment", mask_type)
            frames.append(df)
    if len(frames) < 2:
        return

    cols = [
        "experiment",
        "budget",
        "method",
        "prompt_mae_mean",
        "prompt_mae_std",
        "test_log_loss_mean",
        "test_brier_mean",
        "wasserstein_1_mean",
        "converged_rate",
        "cover_prompt_latent_95_mean",
        "mean_interval_width_latent_95_mean",
        "cover_avg_latent_95_mean",
        "avg_interval_width_latent_95_mean",
        "cover_prompt_predictive_95_mean",
        "mean_interval_width_predictive_95_mean",
        "cover_avg_predictive_95_mean",
        "avg_interval_width_predictive_95_mean",
    ]
    out = pd.concat(frames, ignore_index=True)
    keep = [col for col in cols if col in out.columns]
    compact = out[keep].sort_values(["experiment", "budget", "prompt_mae_mean"])
    compact.to_csv(base / "balanced_vs_random_summary.csv", index=False)


def summary_path(results_dir: str, model_id: str, subject: str, mask_type: str) -> Path:
    return (
        resolve(results_dir)
        / safe_path_component(model_id)
        / subject
        / mask_type
        / f"{mask_type}_summary.csv"
    )


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def safe_path_component(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
