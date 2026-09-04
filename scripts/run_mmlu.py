#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baypire.data.benchmarks.mmlu import build_mmlu_matrix, load_mmlu_long_form
from baypire.experiments.runner import MethodSpec, run_experiment
from baypire.models.bayesian_irt import fit_bayesian_irt
from baypire.models.map_rasch import fit_map_rasch
from baypire.models.naive import predict_naive
from baypire.models.pe_rasch import fit_pe_rasch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BayPIRE recovery on PromptEval MMLU.")
    parser.add_argument("--data", default="data/external/prompteval_mmlu/chunks")
    parser.add_argument("--model-id", default="meta_llama_llama_3_8b")
    parser.add_argument(
        "--subject",
        default="abstract_algebra",
        help="MMLU subject name, or 'all' to pool all loaded subjects.",
    )
    parser.add_argument(
        "--budgets",
        default="200,400,800,1600",
        help="Comma-separated observation budgets. Values <=1 are fractions; values >1 are absolute cell counts.",
    )
    parser.add_argument("--n-masks", type=int, default=10)
    parser.add_argument("--mask-type", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--methods", nargs="+", default=["Avg", "PE-Rasch", "Bayes-MAP", "BayPIRE"])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--prompteval-repo", default="vendor/prompteval")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_path = resolve(args.data)
    df = load_mmlu_long_form(data_path)
    subject = None if str(args.subject).lower() in {"all", "none", "null"} else args.subject
    experiment = build_mmlu_matrix(df, model_id=args.model_id, subject=subject)
    methods = build_methods(args)
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]

    result_df, _ = run_experiment(
        experiment=experiment,
        methods=methods,
        budgets=budgets,
        n_masks=args.n_masks,
        mask_type=args.mask_type,
        base_seed=args.base_seed,
    )

    subject_label = "all_subjects" if subject is None else str(args.subject)
    if args.out:
        out = Path(args.out)
    else:
        out = (
            ROOT
            / "results/mmlu"
            / safe_path_component(args.model_id)
            / safe_path_component(subject_label)
            / safe_path_component(args.mask_type)
            / f"{args.mask_type}.csv"
        )
    if not out.is_absolute():
        out = ROOT / out

    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)

    summary = summarize(result_df)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    print(result_df.to_string(index=False))
    print(f"results={out}")
    print(f"summary={summary_path}")


def build_methods(args: argparse.Namespace) -> list[MethodSpec]:
    selected = {normalise_method_name(name) for name in args.methods}
    methods: list[MethodSpec] = []

    if "Avg" in selected:
        methods.append(MethodSpec("Avg", predict_naive))

    if "PE-Rasch" in selected:
        methods.append(
            MethodSpec(
                "PE-Rasch",
                fit_pe_rasch,
                {"prompteval_repo": resolve(args.prompteval_repo)},
            )
        )

    if "Bayes-MAP" in selected:
        methods.append(MethodSpec("Bayes-MAP", fit_map_rasch))

    if "BayPIRE" in selected:
        methods.append(
            MethodSpec(
                "BayPIRE",
                fit_bayesian_irt,
                {
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "target_accept": args.target_accept,
                    "seed": args.base_seed,
                },
            )
        )

    return methods


def normalise_method_name(name: str) -> str:
    aliases = {
        "M0": "Avg",
        "AVG": "Avg",
        "Avg": "Avg",
        "naive": "Avg",
        "M0_naive": "Avg",
        "PE": "PE-Rasch",
        "PE-Rasch": "PE-Rasch",
        "PE_rasch": "PE-Rasch",
        "prompteval": "PE-Rasch",
        "PromptEval": "PE-Rasch",
        "M1": "Bayes-MAP",
        "Bayes-MAP": "Bayes-MAP",
        "MAP": "Bayes-MAP",
        "M1_map_rasch": "Bayes-MAP",
        "M2": "BayPIRE",
        "BayPIRE": "BayPIRE",
        "Baypire": "BayPIRE",
        "M2_bayesian_irt": "BayPIRE",
    }
    if name not in aliases:
        raise ValueError(f"Unknown method {name!r}. Use Avg, PE-Rasch, Bayes-MAP, BayPIRE.")
    return aliases[name]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["benchmark", "model_id", "budget", "mask_type", "method"]
    aggregations = {}

    for metric in ["prompt_mae", "prompt_rmse", "wasserstein_1", "test_log_loss", "test_brier"]:
        if metric in df.columns:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")

    if "converged" in df.columns:
        df = df.copy()
        df["converged_numeric"] = pd.to_numeric(df["converged"], errors="coerce")
        aggregations["converged_rate"] = ("converged_numeric", "mean")

    interval_prefixes = (
        "cover_prompt_",
        "cover_avg_",
        "mean_interval_width_",
        "median_interval_width_",
        "avg_interval_width_",
    )
    interval_metrics = [
        col
        for col in df.columns
        if col.startswith(interval_prefixes) and pd.api.types.is_numeric_dtype(df[col])
    ]
    for metric in interval_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    return df.groupby(group_cols, as_index=False).agg(**aggregations)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def safe_path_component(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
