#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baypire.data.benchmarks.olympiad import load_olympiad_matrix
from baypire.experiments.runner import MethodSpec, run_experiment
from baypire.models.bayesian_irt import fit_bayesian_irt
from baypire.models.map_rasch import fit_map_rasch
from baypire.models.naive import predict_naive
from baypire.models.pe_rasch import fit_pe_rasch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BayPIRE recovery on Olympiad data.")
    parser.add_argument("--data", default="data/raw/olympiad/olym_math_deepseek_p50/matrices/Y_pilot.pkl")
    parser.add_argument("--model-id", default="deepseek-anthropic")
    parser.add_argument("--budgets", default="0.1,0.2,0.3,0.6")
    parser.add_argument("--n-masks", type=int, default=20)
    parser.add_argument("--mask-type", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    experiment = load_olympiad_matrix(data_path, model_id=args.model_id)
    methods = [
        MethodSpec("M0_naive", predict_naive),
        MethodSpec("PE_rasch", fit_pe_rasch),
        MethodSpec("M1_map_rasch", fit_map_rasch),
        MethodSpec("M2_bayesian_irt", fit_bayesian_irt, {"draws": args.draws, "tune": args.tune, "chains": args.chains}),
    ]
    result_df, _ = run_experiment(
        experiment,
        methods,
        budgets=[float(x) for x in args.budgets.split(",") if x.strip()],
        n_masks=args.n_masks,
        mask_type=args.mask_type,
    )
    out = Path(args.out) if args.out else ROOT / "results/olympiad" / f"{args.model_id}_{args.mask_type}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    print(f"results={out}")


if __name__ == "__main__":
    main()
