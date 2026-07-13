#!/usr/bin/env python3
"""Run the controlled definitional-variable experiment matrix.

Examples:

    python scripts/run_macro_experiment.py --split development
    python scripts/run_macro_experiment.py --split heldout --conditions B0 B1 C1 M2 M4
    python scripts/run_macro_experiment.py --split all --start 0 --end 956
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "B0": {"max_n_gen": 2, "flags": []},
    "B1": {"max_n_gen": 2, "flags": ["--change_of_variables_moves"]},
    "C0": {"max_n_gen": 3, "flags": []},
    "C1": {"max_n_gen": 3, "flags": ["--ac45_moves"]},
    "M1": {
        "max_n_gen": 3,
        "flags": ["--macro_variable_moves", "--macro_gain_policy", "positive"],
    },
    "M2": {
        "max_n_gen": 3,
        "flags": ["--macro_variable_moves", "--macro_gain_policy", "nonnegative"],
    },
    "M3": {
        "max_n_gen": 3,
        "flags": ["--macro_variable_moves", "--macro_gain_policy", "relaxed"],
    },
    "M4": {
        "max_n_gen": 3,
        "flags": [
            "--macro_variable_moves",
            "--macro_gain_policy",
            "nonnegative",
            "--change_of_variables_moves",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="1190MS")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=956)
    parser.add_argument(
        "--split", choices=["development", "heldout", "all"], default="development"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=sorted(CONDITIONS),
    )
    parser.add_argument("--out_dir", default="results/macro_experiment")
    parser.add_argument("--max_nodes", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=24)
    parser.add_argument("--max_excess_length", type=int, default=98)
    parser.add_argument("--macro_min_word_length", type=int, default=2)
    parser.add_argument("--macro_max_word_length", type=int, default=5)
    parser.add_argument("--macro_min_occurrences", type=int, default=2)
    parser.add_argument("--macro_top_k", type=int, default=3)
    parser.add_argument("--macro_relaxed_slack", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose_solver", action="store_true")
    return parser.parse_args()


def split_args(split: str) -> list[str]:
    if split == "development":
        return ["--index_modulus", "5", "--index_remainder", "0"]
    if split == "heldout":
        return [
            "--index_modulus",
            "5",
            "--index_remainders",
            "1",
            "2",
            "3",
            "4",
        ]
    return []


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")
    if args.max_nodes <= 0 or args.max_length <= 0:
        raise SystemExit("--max_nodes and --max_length must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    solver = repo_root / "scripts" / "greedy_search_n.py"
    out_dir = Path(args.out_dir).expanduser().resolve() / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": args.dataset,
        "start": args.start,
        "end": args.end,
        "split": args.split,
        "conditions": args.conditions,
        "max_nodes": args.max_nodes,
        "max_length": args.max_length,
        "max_excess_length": args.max_excess_length,
        "macro_min_word_length": args.macro_min_word_length,
        "macro_max_word_length": args.macro_max_word_length,
        "macro_min_occurrences": args.macro_min_occurrences,
        "macro_top_k": args.macro_top_k,
        "macro_relaxed_slack": args.macro_relaxed_slack,
        "python": sys.executable,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for name in args.conditions:
        output = out_dir / f"{name}.jsonl"
        if output.exists() and not args.overwrite:
            raise SystemExit(
                f"{output} already exists; use --overwrite or another --out_dir"
            )
        condition = CONDITIONS[name]
        command = [
            sys.executable,
            str(solver),
            "--dataset",
            args.dataset,
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--max_n_gen",
            str(condition["max_n_gen"]),
            "--max_nodes",
            str(args.max_nodes),
            "--max_length",
            str(args.max_length),
            "--max_excess_length",
            str(args.max_excess_length),
            "--priority_metric",
            "excess",
            "--macro_min_word_length",
            str(args.macro_min_word_length),
            "--macro_max_word_length",
            str(args.macro_max_word_length),
            "--macro_min_occurrences",
            str(args.macro_min_occurrences),
            "--macro_top_k",
            str(args.macro_top_k),
            "--macro_relaxed_slack",
            str(args.macro_relaxed_slack),
            "--out_jsonl",
            str(output),
            *split_args(args.split),
            *condition["flags"],
        ]
        if not args.verbose_solver:
            command.append("--quiet")
        print(f"RUN {name}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=repo_root, check=True)

    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
