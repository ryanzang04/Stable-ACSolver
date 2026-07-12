"""Summarize change-of-variables substitutions from greedy search paths.

Input is a JSONL file produced by:

    python scripts/greedy_search_n.py --dataset 1190MS --all --compare_cov \
        --compare_cov_paths --out_jsonl results/cov_compare_1190MS_paths.jsonl

The script extracts every COV action from solved COV paths, reconstructs the
pre-move presentation and selected cyclic word W, and reports frequencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


GENERATOR_LETTERS = "xyzuvwabcdefghijklmnopqrst"


def num_substitution_branches(max_n_gen: int) -> int:
    return 2 if max_n_gen == 2 else max_n_gen * (max_n_gen - 1)


def num_change_of_variables_branches(max_n_gen: int) -> int:
    return max_n_gen * max_n_gen


def is_cov_action(action: list[int], max_n_gen: int) -> bool:
    branch = int(action[0])
    start = num_substitution_branches(max_n_gen)
    end = start + num_change_of_variables_branches(max_n_gen)
    return start <= branch < end


def decode_cov_action(action: list[int], max_n_gen: int) -> dict:
    branch, z_inverse, z_start, z_len_code = map(int, action)
    cov_branch = branch - num_substitution_branches(max_n_gen)
    return {
        "remove_gen": cov_branch // max_n_gen,
        "iso_relator": cov_branch % max_n_gen,
        "z_inverse": z_inverse,
        "z_start": z_start,
        "z_len": z_len_code + 1,
    }


def inverse_word(word: Iterable[int]) -> tuple[int, ...]:
    return tuple(-x for x in reversed(tuple(word)))


def cyclic_split(word: tuple[int, ...], start: int, length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = len(word)
    if n == 0:
        return (), ()
    rotated = tuple(word[(start + i) % n] for i in range(n))
    return rotated[:length], rotated[length:]


def format_letter(code: int) -> str:
    idx = abs(code) - 1
    if 0 <= idx < len(GENERATOR_LETTERS):
        letter = GENERATOR_LETTERS[idx]
        return letter if code > 0 else letter.upper()
    return f"g{code}"


def format_word(word: Iterable[int]) -> str:
    values = tuple(int(v) for v in word)
    return "".join(format_letter(v) for v in values) if values else "1"


def format_presentation(presentation: list[list[int]]) -> str:
    return " | ".join(format_word(word) for word in presentation)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def cov_result_from_row(row: dict) -> dict | None:
    if "with_cov_result" in row:
        return row["with_cov_result"]
    if "path" in row and "actions" in row:
        return row
    return None


def extract_events(row: dict, max_n_gen: int) -> list[dict]:
    result = cov_result_from_row(row)
    if result is None:
        return []
    if not row.get("with_cov_solved", result.get("solved", False)):
        return []

    path = result.get("path") or []
    actions = result.get("actions") or []
    idx = row.get("idx", result.get("idx", -1))
    events = []
    cov_ordinal = 0

    for step, action in enumerate(actions):
        if not is_cov_action(action, max_n_gen):
            continue
        if step >= len(path):
            continue
        pre = path[step]
        post = path[step + 1] if step + 1 < len(path) else []
        decoded = decode_cov_action(action, max_n_gen)
        iso_relator = decoded["iso_relator"]
        if iso_relator >= len(pre):
            continue

        iso_word = tuple(int(v) for v in pre[iso_relator])
        window, complement = cyclic_split(
            iso_word, decoded["z_start"], decoded["z_len"]
        )
        z_word = inverse_word(window) if decoded["z_inverse"] else window
        old_code = decoded["remove_gen"] + 1
        old_count_in_complement = sum(1 for v in complement if abs(v) == old_code)

        cov_ordinal += 1
        event = {
            "idx": idx,
            "step": step,
            "cov_ordinal": cov_ordinal,
            "path_length": result.get("path_length", len(actions)),
            "remove_gen": decoded["remove_gen"],
            "removed_generator": format_letter(old_code),
            "iso_relator": iso_relator,
            "z_inverse": decoded["z_inverse"],
            "z_start": decoded["z_start"],
            "z_len": decoded["z_len"],
            "window": format_word(window),
            "z_word": format_word(z_word),
            "complement": format_word(complement),
            "old_count_in_complement": old_count_in_complement,
            "pre_presentation": format_presentation(pre),
            "post_presentation": format_presentation(post),
            "pre_total_len": sum(len(word) for word in pre),
            "post_total_len": sum(len(word) for word in post),
            "action": json.dumps(action),
        }
        event["length_delta"] = event["post_total_len"] - event["pre_total_len"]
        event["signature"] = (
            f"remove={event['removed_generator']} "
            f"iso=R{iso_relator} z={event['z_word']}"
        )
        events.append(event)
    return events


def print_counter(title: str, counter: Counter, top: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, count in counter.most_common(top):
        print(f"{count:6d}  {key}")


def write_counter_csv(path: Path, counter: Counter, key_name: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, "count"])
        for key, count in counter.most_common():
            writer.writerow([key, count])


def write_events_csv(path: Path, events: list[dict]) -> None:
    if not events:
        return
    fields = [
        "idx",
        "step",
        "cov_ordinal",
        "path_length",
        "removed_generator",
        "remove_gen",
        "iso_relator",
        "z_inverse",
        "z_start",
        "z_len",
        "window",
        "z_word",
        "complement",
        "old_count_in_complement",
        "pre_total_len",
        "post_total_len",
        "length_delta",
        "signature",
        "pre_presentation",
        "post_presentation",
        "action",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze COV variable substitutions from solved paths."
    )
    parser.add_argument("--jsonl", required=True, help="comparison JSONL with paths")
    parser.add_argument("--max_n_gen", type=int, default=2)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="optional directory for CSV outputs",
    )
    args = parser.parse_args()

    rows = list(load_jsonl(Path(args.jsonl)))
    missing_paths = 0
    cov_solved_rows = 0
    solved_with_cov_events = 0
    all_events: list[dict] = []
    first_events: list[dict] = []

    for row in rows:
        if row.get("with_cov_solved", row.get("solved", False)):
            cov_solved_rows += 1
        result = cov_result_from_row(row)
        if result is None and row.get("with_cov_solved", False):
            missing_paths += 1
            continue
        events = extract_events(row, args.max_n_gen)
        if events:
            solved_with_cov_events += 1
            all_events.extend(events)
            first_events.append(events[0])

    no_cov_event_solved = cov_solved_rows - solved_with_cov_events - missing_paths

    print(f"rows: {len(rows)}")
    print(f"cov_solved_rows: {cov_solved_rows}")
    print(f"cov_solved_rows_missing_paths: {missing_paths}")
    print(f"cov_solved_rows_with_at_least_one_cov_action: {solved_with_cov_events}")
    print(f"cov_solved_rows_with_no_cov_action: {no_cov_event_solved}")
    print(f"total_cov_actions_in_solved_paths: {len(all_events)}")

    counters = {
        "first_z_word": Counter(e["z_word"] for e in first_events),
        "all_z_word": Counter(e["z_word"] for e in all_events),
        "first_window": Counter(e["window"] for e in first_events),
        "all_window": Counter(e["window"] for e in all_events),
        "first_pre_presentation": Counter(e["pre_presentation"] for e in first_events),
        "all_pre_presentation": Counter(e["pre_presentation"] for e in all_events),
        "first_signature": Counter(e["signature"] for e in first_events),
        "all_signature": Counter(e["signature"] for e in all_events),
        "z_len": Counter(e["z_len"] for e in all_events),
        "iso_relator": Counter(e["iso_relator"] for e in all_events),
        "removed_generator": Counter(e["removed_generator"] for e in all_events),
        "length_delta": Counter(e["length_delta"] for e in all_events),
    }

    print_counter("First COV: z word frequencies", counters["first_z_word"], args.top)
    print_counter("All COV: z word frequencies", counters["all_z_word"], args.top)
    print_counter(
        "First COV: pre-presentation frequencies",
        counters["first_pre_presentation"],
        args.top,
    )
    print_counter(
        "All COV: pre-presentation frequencies",
        counters["all_pre_presentation"],
        args.top,
    )
    print_counter("All COV: signature frequencies", counters["all_signature"], args.top)
    print_counter("All COV: z length frequencies", counters["z_len"], args.top)
    print_counter("All COV: length-delta frequencies", counters["length_delta"], args.top)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_events_csv(out_dir / "cov_events.csv", all_events)
        for name, counter in counters.items():
            write_counter_csv(out_dir / f"{name}.csv", counter, name)
        print(f"\nwrote CSV summaries to {out_dir}")


if __name__ == "__main__":
    main()
