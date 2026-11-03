#!/usr/bin/env python3
"""MS640 CoV-gate best-first experiment; run beside greedy_search_n.py."""
import argparse
import ast
import hashlib
import heapq
import json
import os
import sys
import time
from pathlib import Path

import greedy_search_n as g

GATES = ("no_short_sub", "short_relator", "unbalanced", "long_presentation", "plateau100", "periodic10")


def search(flat, gate, args, apply_batch):
    solver = g.GreedyNSolver(flat, 2, 2, 24, args.max_nodes, 48, None, True, False,
                             args.chunk_size, False, apply_batch=apply_batch)
    root = solver.initial_key
    parents = {root: (None, None)}
    flats = {root: solver.initial_flat}
    queue = [(g.state_total_length(root), 0, 0, root)]
    seq = visited = checks = openings = cov_actions = cov_changed = cov_shorter = cov_shorter_unique = openings_shorter = 0
    best_length = g.state_total_length(root)
    last_improved = 0
    start = time.monotonic()
    last_report = start

    def expand(key, depth, actions, family):
        nonlocal seq, cov_changed, cov_shorter, cov_shorter_unique
        current_length = g.state_total_length(key)
        shorter_keys = set()
        for action, next_flat, next_n in g.apply_actions_chunked(
                apply_batch, flats[key], key[0], actions, args.chunk_size):
            next_key, canonical_flat = g.canonicalize_flat(next_flat, next_n, 2, 24)
            length = g.state_total_length(next_key)
            if next_key == key or length > 48:
                continue
            if family == "cov":
                cov_changed += 1
                if length < current_length:
                    cov_shorter += 1
                    shorter_keys.add(next_key)
            elif length < current_length:
                shorter_keys.add(next_key)
            if next_key in parents:
                continue
            seq += 1
            parents[next_key] = (key, action)
            flats[next_key] = canonical_flat
            heapq.heappush(queue, (length, depth + 1, seq, next_key))
        if family == "cov":
            cov_shorter_unique += len(shorter_keys)
        return bool(shorter_keys)

    while queue and visited < args.max_nodes:
        length, depth, _, key = heapq.heappop(queue)
        visited += 1
        if g.is_trivial_key(key):
            keys, actions = solver._reconstruct(key, parents)
            return result(True, "solved", visited, parents, checks, openings, cov_actions,
                          cov_changed, cov_shorter, cov_shorter_unique, openings_shorter,
                          start, keys, actions, best_length)
        if length < best_length:
            best_length, last_improved = length, visited
        subs = list(g.iter_substitution_actions(key, 2))
        sub_shortens = expand(key, depth, subs, "sub")
        checks += 1
        lengths = [len(word) for word in key[1]]
        should_open = {
            "no_short_sub": not sub_shortens,
            "short_relator": min(lengths) <= 3,
            "unbalanced": max(lengths) >= 2 * min(lengths),
            "long_presentation": length >= 16,
            "plateau100": visited - last_improved >= 100,
            "periodic10": visited % 10 == 0,
        }[gate]
        if should_open:
            openings += 1
            moves = list(g.iter_change_of_variables_actions(key, 2, 1, None, 1))
            cov_actions += len(moves)
            openings_shorter += int(expand(key, depth, moves, "cov"))
        now = time.monotonic()
        if now - last_report >= args.progress_seconds:
            print(f"progress gate={gate} visited={visited} seen={len(parents)} "
                  f"openings={openings} elapsed={now-start:.1f}s", flush=True)
            last_report = now
    return result(False, "budget" if queue else "frontier_exhausted", visited, parents,
                  checks, openings, cov_actions, cov_changed, cov_shorter, cov_shorter_unique,
                  openings_shorter, start, [], [], best_length)


def result(solved, reason, visited, parents, checks, openings, actions, changed, shorter,
           shorter_unique, openings_shorter, started, keys, path_actions, best_length):
    return {"solved": solved, "termination": reason, "nodes_visited": visited,
            "states_seen": len(parents), "elapsed_sec": round(time.monotonic()-started, 3),
            "path_length": len(path_actions) if solved else None,
            "path": [g.presentation_payload(key) for key in keys],
            "actions": [g.action_payload(action) for action in path_actions],
            "gate_checks": checks, "gate_openings": openings,
            "cov_actions_enumerated": actions, "cov_changed_in_cap_actions": changed,
            "cov_shortening_actions": shorter, "cov_shortening_unique_per_parent_sum": shorter_unique,
            "gate_openings_with_shortening": openings_shorter, "best_popped_length": best_length}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gate", choices=GATES, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=640)
    p.add_argument("--max-nodes", type=int, default=1_000_000)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--progress-seconds", type=float, default=60)
    a = p.parse_args()
    if a.start < 0 or a.end <= a.start or a.max_nodes < 1 or a.chunk_size < 1 or a.progress_seconds <= 0:
        p.error("invalid range, budget, chunk size or progress interval")
    source = Path(a.input).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    lines = source.decode().splitlines()
    if a.end > len(lines):
        p.error(f"end exceeds {len(lines)} input rows")
    config = {"dataset_sha256": digest, "gate": a.gate, "max_nodes": a.max_nodes,
              "max_length": 24, "max_total_length": 48, "chunk_size": a.chunk_size,
              "cov_window_min_length": 1, "cov_window_relator_margin": 1,
              "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "search_source_sha256": hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),
              "schema_version": 1}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        with out.open() as f:
            for number, line in enumerate(f, 1):
                try:
                    old = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON at {out}:{number}; preserve and repair it before resuming: {exc}")
                if old["config"] != config:
                    raise SystemExit(f"Configuration mismatch in {out}:{number}; choose a new output file")
                if old["idx"] in completed:
                    raise SystemExit(f"Duplicate row {old['idx']} in {out}")
                completed.add(old["idx"])
    params = g.MoveParams(n_gen=2, max_n_gen=2, max_length=24, max_steps_in_episode=a.max_nodes)
    apply_batch = g.build_apply_batch(params, True, False)
    print(f"gate={a.gate} rows={a.start}:{a.end} resume_completed={len(completed)} config={json.dumps(config)}", flush=True)
    with out.open("a", buffering=1) as f:
        for idx in range(a.start, a.end):
            if idx in completed:
                continue
            raw = ast.literal_eval(lines[idx])
            if len(raw) != 48 or any(type(x) is not int or x not in (-2, -1, 0, 1, 2) for x in raw):
                raise ValueError(f"invalid 48-entry rank-two row {idx}")
            flat, _, _ = g.normalize_dataset_flat(raw, 2, 2, 24)
            print(f"start gate={a.gate} idx={idx}", flush=True)
            data = search(flat, a.gate, a, apply_batch)
            data.update(idx=idx, config=config)
            f.write(json.dumps(data, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"done gate={a.gate} idx={idx} solved={data['solved']} nodes={data['nodes_visited']} seen={data['states_seen']} path={data['path_length']} openings={data['gate_openings']} shortened={data['cov_shortening_actions']}", flush=True)


if __name__ == "__main__":
    main()
