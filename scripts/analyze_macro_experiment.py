#!/usr/bin/env python3
"""Summarize JSONL files produced by run_macro_experiment.py."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx in rows:
                raise ValueError(f"duplicate idx {idx} in {path}:{line_number}")
            rows[idx] = row
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--baseline", default="B0")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no JSONL files found in {results_dir}")
    conditions = {path.stem: load_jsonl(path) for path in files}
    if args.baseline not in conditions:
        raise SystemExit(f"baseline {args.baseline!r} not found in {results_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else results_dir / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, rows_by_idx in conditions.items():
        rows = list(rows_by_idx.values())
        solved_rows = [row for row in rows if row["solved"]]
        summary_rows.append(
            {
                "condition": name,
                "instances": len(rows),
                "solved": len(solved_rows),
                "solve_rate": len(solved_rows) / max(len(rows), 1),
                "mean_nodes_all": mean([row["nodes_visited"] for row in rows]),
                "median_nodes_all": median([row["nodes_visited"] for row in rows]),
                "mean_seen_all": mean([row["seen_count"] for row in rows]),
                "mean_runtime_all": mean([row["elapsed_sec"] for row in rows]),
                "mean_depth_solved": mean(
                    [row["path_length"] for row in solved_rows]
                ),
                "median_depth_solved": median(
                    [row["path_length"] for row in solved_rows]
                ),
                "solutions_using_macro": sum(
                    row.get("macro_moves_in_solution", 0) > 0 for row in solved_rows
                ),
                "mean_effective_branching": mean(
                    [row.get("effective_branching_factor", 0.0) for row in rows]
                ),
                "mean_macro_candidates_generated": mean(
                    [row.get("macro_candidates_generated", 0) for row in rows]
                ),
            }
        )

    summary_fields = list(summary_rows[0])
    write_csv(out_dir / "condition_summary.csv", summary_rows, summary_fields)

    baseline = conditions[args.baseline]
    pairwise_rows = []
    instance_rows = []
    for name, candidate in conditions.items():
        common = sorted(set(baseline) & set(candidate))
        improved = []
        worsened = []
        both = []
        neither = []
        depth_deltas = []
        node_deltas = []
        for idx in common:
            base_row = baseline[idx]
            candidate_row = candidate[idx]
            base_solved = bool(base_row["solved"])
            candidate_solved = bool(candidate_row["solved"])
            if candidate_solved and not base_solved:
                relation = "improved"
                improved.append(idx)
            elif base_solved and not candidate_solved:
                relation = "worsened"
                worsened.append(idx)
            elif base_solved and candidate_solved:
                relation = "both"
                both.append(idx)
                depth_deltas.append(
                    candidate_row["path_length"] - base_row["path_length"]
                )
                node_deltas.append(
                    candidate_row["nodes_visited"] - base_row["nodes_visited"]
                )
            else:
                relation = "neither"
                neither.append(idx)
            instance_rows.append(
                {
                    "condition": name,
                    "idx": idx,
                    "relation_to_baseline": relation,
                    "baseline_solved": base_solved,
                    "condition_solved": candidate_solved,
                    "baseline_depth": base_row["path_length"],
                    "condition_depth": candidate_row["path_length"],
                    "baseline_nodes": base_row["nodes_visited"],
                    "condition_nodes": candidate_row["nodes_visited"],
                    "macro_moves_in_solution": candidate_row.get(
                        "macro_moves_in_solution", 0
                    ),
                }
            )

        pairwise_rows.append(
            {
                "condition": name,
                "common_instances": len(common),
                "improved_count": len(improved),
                "worsened_count": len(worsened),
                "net_solved_gain": len(improved) - len(worsened),
                "both_solved_count": len(both),
                "neither_count": len(neither),
                "mean_depth_delta_both": mean(depth_deltas),
                "median_depth_delta_both": median(depth_deltas),
                "mean_node_delta_both": mean(node_deltas),
                "improved_indices": " ".join(map(str, improved)),
                "worsened_indices": " ".join(map(str, worsened)),
            }
        )

    write_csv(
        out_dir / "pairwise_vs_baseline.csv",
        pairwise_rows,
        list(pairwise_rows[0]),
    )
    write_csv(
        out_dir / "instance_comparison.csv",
        instance_rows,
        list(instance_rows[0]),
    )

    event_rows = []
    word_counts: Counter[tuple[str, str]] = Counter()
    for name, rows_by_idx in conditions.items():
        for idx, row in rows_by_idx.items():
            for event in row.get("macro_events", []):
                event_rows.append({"condition": name, "idx": idx, **event})
                word_counts[(name, event["word_text"])] += 1
    if event_rows:
        event_fields = []
        for row in event_rows:
            for field in row:
                if field not in event_fields:
                    event_fields.append(field)
        write_csv(out_dir / "macro_events.csv", event_rows, event_fields)
        frequency_rows = [
            {"condition": condition, "word": word, "frequency": frequency}
            for (condition, word), frequency in sorted(
                word_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
            )
        ]
        write_csv(
            out_dir / "macro_word_frequency.csv",
            frequency_rows,
            ["condition", "word", "frequency"],
        )

    if "B0" in conditions and "C0" in conditions:
        invariant_mismatches = []
        for idx in sorted(set(conditions["B0"]) & set(conditions["C0"])):
            rank_two = conditions["B0"][idx]
            capacity_three = conditions["C0"][idx]
            compared_fields = [
                "solved",
                "nodes_visited",
                "seen_count",
                "path_length",
                "path",
            ]
            differing = [
                field
                for field in compared_fields
                if rank_two.get(field) != capacity_three.get(field)
            ]
            if differing:
                invariant_mismatches.append(
                    {"idx": idx, "differing_fields": " ".join(differing)}
                )
        write_csv(
            out_dir / "capacity_invariance.csv",
            invariant_mismatches,
            ["idx", "differing_fields"],
        )
        print(
            "B0/C0 capacity invariance mismatches: "
            f"{len(invariant_mismatches)}"
        )

    print(f"Wrote analysis to {out_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']}: solved={row['solved']}/{row['instances']} "
            f"rate={row['solve_rate']:.3f} "
            f"mean_nodes={row['mean_nodes_all']:.1f} "
            f"macro_solutions={row['solutions_using_macro']}"
        )


if __name__ == "__main__":
    main()
