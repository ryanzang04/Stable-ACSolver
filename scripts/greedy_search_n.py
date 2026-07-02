"""Greedy best-first search for n-generator Andrews-Curtis presentations.

This is a script version of the idea in ``greedy_search.ipynb``:

* canonicalize presentations up to cyclic rotation, inversion, and relator order;
* prioritize states by total active relator length;
* expand substitution supermoves with a boundary cancellation.

Unlike the notebook, this script works with a fixed generator capacity
``max_n_gen`` and can opt into the stable moves implemented by ``envs.ac_moves``:
change-of-variables, AC4 add-generator, and AC5 delete-generator.

Examples, run from the repository root:

    python scripts/greedy_search_n.py --relators XyyyxYYYY XYXyxy --max_nodes 10000
    python scripts/greedy_search_n.py --dataset 1190MS --idx 0 --max_nodes 10000
    python scripts/greedy_search_n.py --relators xyxYXY xxxYYYY \
        --change_of_variables_moves --max_n_gen 2 --show_path
"""

from __future__ import annotations

import argparse
import ast
import gzip
import heapq
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Sequence
import types

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    from gymnax.environments import environment as _gymnax_environment  # noqa: F401
except ModuleNotFoundError:
    # envs.ac_moves imports envs.environment only for its EnvParams type.
    # The greedy script does not instantiate a gymnax Environment, so a narrow
    # shim keeps the move functions usable in lightweight local environments.
    gymnax_mod = types.ModuleType("gymnax")
    environments_mod = types.ModuleType("gymnax.environments")
    environment_mod = types.ModuleType("gymnax.environments.environment")

    class _ShimEnvParams:
        pass

    class _ShimEnvState:
        pass

    class _ShimEnvironment:
        pass

    environment_mod.EnvParams = _ShimEnvParams
    environment_mod.EnvState = _ShimEnvState
    environment_mod.Environment = _ShimEnvironment
    environments_mod.environment = environment_mod
    gymnax_mod.environments = environments_mod
    sys.modules.setdefault("gymnax", gymnax_mod)
    sys.modules.setdefault("gymnax.environments", environments_mod)
    sys.modules.setdefault("gymnax.environments.environment", environment_mod)

import jax
import jax.numpy as jnp
import numpy as np

from envs.ac_moves import setup_ac_actions
from envs.utils import (
    add_generator_branch,
    change_of_variables_branch,
    change_presentation_shape,
    convert_relator_list_to_presentation,
    delete_generator_branch,
    encode_action,
    num_change_of_variables_branches,
    num_substitution_branches,
)


GENERATOR_LETTERS = "xyzuvwabcdefghijklmnopqrst"
LETTER_TO_CODE = {}
for idx, letter in enumerate(GENERATOR_LETTERS, start=1):
    LETTER_TO_CODE[letter] = idx
    LETTER_TO_CODE[letter.upper()] = -idx


StateKey = tuple[int, tuple[tuple[int, ...], ...]]
Action = tuple[int, int, int, int]


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    nodes_visited: int
    elapsed_sec: float
    final_key: StateKey | None
    path_keys: list[StateKey]
    path_actions: list[Action]
    seen_count: int


@dataclass(frozen=True)
class MoveParams:
    n_gen: int = 2
    max_n_gen: int = 2
    max_length: int = 24
    max_steps_in_episode: int = 200


def parse_word(text: str) -> list[int]:
    """Parse either a compact word like ``xyX`` or a Python list of ints."""
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        values = ast.literal_eval(text)
        if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
            raise ValueError(f"expected a list of ints, got {text!r}")
        return [v for v in values if v != 0]

    out = []
    for ch in text:
        if ch not in LETTER_TO_CODE:
            raise ValueError(
                f"unsupported generator letter {ch!r}; use letters from "
                f"{GENERATOR_LETTERS!r} or pass a Python int list"
            )
        out.append(LETTER_TO_CODE[ch])
    return out


def format_word(word: Sequence[int]) -> str:
    if not word:
        return "1"
    parts = []
    for code in word:
        abs_code = abs(int(code))
        if abs_code <= len(GENERATOR_LETTERS):
            letter = GENERATOR_LETTERS[abs_code - 1]
            parts.append(letter if code > 0 else letter.upper())
        else:
            parts.append(f"x{abs_code}" if code > 0 else f"X{abs_code}")
    return "".join(parts)


def state_total_length(key: StateKey) -> int:
    return sum(len(word) for word in key[1])


def letter_order(code: int, active_n_gen: int) -> int:
    """Notebook-compatible order generalized from Y < y < X < x."""
    abs_code = abs(int(code))
    return 2 * (active_n_gen - abs_code) + (1 if code > 0 else 0)


def word_order_key(word: Sequence[int], active_n_gen: int) -> tuple[int, ...]:
    return tuple(letter_order(code, active_n_gen) for code in word)


def inverse_word(word: Sequence[int]) -> tuple[int, ...]:
    return tuple(-int(code) for code in reversed(word))


def free_reduce(word: Sequence[int]) -> tuple[int, ...]:
    stack: list[int] = []
    for code in word:
        code = int(code)
        if code == 0:
            continue
        if stack and stack[-1] == -code:
            stack.pop()
        else:
            stack.append(code)
    return tuple(stack)


def cyclic_reduce(word: Sequence[int]) -> tuple[int, ...]:
    reduced = list(free_reduce(word))
    while len(reduced) > 1 and reduced[0] == -reduced[-1]:
        reduced = list(free_reduce(reduced[1:-1]))
    return tuple(reduced)


def minimal_rotation(word: Sequence[int], active_n_gen: int) -> tuple[int, ...]:
    word = tuple(word)
    if len(word) <= 1:
        return word
    rotations = (word[i:] + word[:i] for i in range(len(word)))
    return min(rotations, key=lambda w: word_order_key(w, active_n_gen))


def canonical_word(word: Sequence[int], active_n_gen: int) -> tuple[int, ...]:
    reduced = cyclic_reduce(word)
    if not reduced:
        return ()
    normal = minimal_rotation(reduced, active_n_gen)
    inverted = minimal_rotation(inverse_word(reduced), active_n_gen)
    return min(normal, inverted, key=lambda w: word_order_key(w, active_n_gen))


def canonicalize_relators(
    relators: Sequence[Sequence[int]], active_n_gen: int
) -> tuple[tuple[int, ...], ...]:
    canon = [canonical_word(word, active_n_gen) for word in relators[:active_n_gen]]
    return tuple(
        sorted(canon, key=lambda w: (len(w), word_order_key(w, active_n_gen)))
    )


def flat_to_relators(
    flat: Sequence[int], active_n_gen: int, max_length: int
) -> list[tuple[int, ...]]:
    arr = np.asarray(flat)
    relators = []
    for idx in range(active_n_gen):
        slot = arr[idx * max_length : (idx + 1) * max_length]
        relators.append(tuple(int(v) for v in slot if int(v) != 0))
    return relators


def key_to_flat(key: StateKey, max_n_gen: int, max_length: int) -> np.ndarray:
    active_n_gen, relators = key
    if active_n_gen > max_n_gen:
        raise ValueError("active_n_gen exceeds max_n_gen")
    flat: list[int] = []
    for word in relators:
        if len(word) > max_length:
            raise ValueError("relator exceeds max_length")
        flat.extend(word)
        flat.extend([0] * (max_length - len(word)))
    for _ in range(max_n_gen - active_n_gen):
        flat.extend([0] * max_length)
    return np.asarray(flat, dtype=np.int8)


def canonicalize_flat(
    flat: Sequence[int], active_n_gen: int, max_n_gen: int, max_length: int
) -> tuple[StateKey, np.ndarray]:
    relators = flat_to_relators(flat, active_n_gen, max_length)
    canon_relators = canonicalize_relators(relators, active_n_gen)
    key: StateKey = (int(active_n_gen), canon_relators)
    return key, key_to_flat(key, max_n_gen, max_length)


def is_trivial_key(key: StateKey) -> bool:
    active_n_gen, relators = key
    if len(relators) != active_n_gen:
        return False
    found = set()
    for word in relators:
        if len(word) != 1:
            return False
        found.add(abs(word[0]))
    return found == set(range(1, active_n_gen + 1))


def pair_to_branch(target: int, source: int, max_n_gen: int) -> int:
    if target == source:
        raise ValueError("target and source must differ")
    if max_n_gen == 2:
        return target
    source_code = source if source < target else source - 1
    return target * (max_n_gen - 1) + source_code


def branch_to_pair(branch: int, max_n_gen: int) -> tuple[int, int]:
    if max_n_gen == 2:
        return branch, 1 - branch
    target = branch // (max_n_gen - 1)
    source = branch % (max_n_gen - 1)
    if source >= target:
        source += 1
    return target, source


def iter_substitution_actions(key: StateKey, max_n_gen: int) -> Iterator[Action]:
    active_n_gen, relators = key
    if active_n_gen < 2:
        return

    if max_n_gen == 2:
        target_source_pairs = [(0, 1)]
        replacement_branches = [0, 1]
    else:
        target_source_pairs = [
            (target, source)
            for target in range(active_n_gen)
            for source in range(active_n_gen)
            if target != source
        ]
        replacement_branches = []

    for target, source in target_source_pairs:
        target_word = relators[target]
        source_word = relators[source]
        if not target_word or not source_word:
            continue
        branches = (
            replacement_branches
            if max_n_gen == 2
            else [pair_to_branch(target, source, max_n_gen)]
        )
        for target_pos, target_code in enumerate(target_word):
            k_target = target_pos + 1
            for source_pos, source_code in enumerate(source_word):
                if target_code == -source_code:
                    for branch in branches:
                        yield (branch, 0, k_target, source_pos)
                if target_code == source_code:
                    for branch in branches:
                        yield (branch, 1, k_target, -source_pos - 1)


def cyclic_subword_complement(
    word: Sequence[int], start: int, length: int
) -> tuple[int, ...]:
    n = len(word)
    rotated = tuple(word[(start + i) % n] for i in range(n))
    return rotated[length:]


def iter_change_of_variables_actions(key: StateKey, max_n_gen: int) -> Iterator[Action]:
    active_n_gen, relators = key
    for remove_gen in range(active_n_gen):
        old_code = remove_gen + 1
        for iso_relator, word in enumerate(relators):
            rel_len = len(word)
            if rel_len <= 1:
                continue
            branch = change_of_variables_branch(remove_gen, iso_relator, max_n_gen)
            for z_start in range(rel_len):
                for z_len in range(1, rel_len):
                    complement = cyclic_subword_complement(word, z_start, z_len)
                    if sum(1 for code in complement if abs(code) == old_code) != 1:
                        continue
                    for z_inverse in (0, 1):
                        yield (branch, z_inverse, z_start, z_len - 1)


def can_delete_generator(key: StateKey, delete_gen: int) -> bool:
    active_n_gen, relators = key
    target_code = delete_gen + 1
    trivial_slots = 0
    contains_slots = 0
    for word in relators[:active_n_gen]:
        contains = any(abs(code) == target_code for code in word)
        if contains:
            contains_slots += 1
        if len(word) == 1 and abs(word[0]) == target_code:
            trivial_slots += 1
    return trivial_slots >= 1 and contains_slots == 1


def iter_ac45_actions(
    key: StateKey, max_n_gen: int, change_of_variables_moves: bool
) -> Iterator[Action]:
    active_n_gen, _ = key
    if active_n_gen < max_n_gen:
        yield (add_generator_branch(max_n_gen, change_of_variables_moves), 0, 0, 0)

    delete_branch = delete_generator_branch(max_n_gen, change_of_variables_moves)
    for delete_gen in range(active_n_gen):
        if can_delete_generator(key, delete_gen):
            yield (delete_branch, delete_gen, 0, 0)


def iter_actions(
    key: StateKey,
    max_n_gen: int,
    change_of_variables_moves: bool,
    ac45_moves: bool,
) -> Iterator[Action]:
    yield from iter_substitution_actions(key, max_n_gen)
    if change_of_variables_moves:
        yield from iter_change_of_variables_actions(key, max_n_gen)
    if ac45_moves:
        yield from iter_ac45_actions(key, max_n_gen, change_of_variables_moves)


def build_apply_batch(
    params: MoveParams, change_of_variables_moves: bool, ac45_moves: bool
):
    actions = setup_ac_actions(
        params,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
    )

    @partial(jax.jit, static_argnames=())
    def apply_batch(x, active_n_gen, action_batch):
        def apply_one(action):
            return jax.lax.switch(
                action[0], actions, x, active_n_gen, action[1:]
            )

        return jax.vmap(apply_one)(action_batch)

    return apply_batch


def apply_actions_chunked(
    apply_batch,
    flat: np.ndarray,
    active_n_gen: int,
    actions: Sequence[Action],
    chunk_size: int,
) -> Iterator[tuple[Action, np.ndarray, int]]:
    if not actions:
        return
    x_jax = jnp.asarray(flat, dtype=jnp.int8)
    n_jax = jnp.asarray(active_n_gen, dtype=jnp.int32)

    for start in range(0, len(actions), chunk_size):
        chunk = list(actions[start : start + chunk_size])
        real_size = len(chunk)
        if real_size < chunk_size:
            chunk.extend([chunk[0]] * (chunk_size - real_size))
        action_batch = jnp.asarray(chunk, dtype=jnp.int32)
        next_x, next_n_gen = apply_batch(x_jax, n_jax, action_batch)
        next_x_np = np.asarray(next_x[:real_size], dtype=np.int8)
        next_n_np = np.asarray(next_n_gen[:real_size], dtype=np.int32)
        for offset in range(real_size):
            yield actions[start + offset], next_x_np[offset], int(next_n_np[offset])


class GreedyNSolver:
    def __init__(
        self,
        initial_flat: np.ndarray,
        n_gen: int,
        max_n_gen: int,
        max_length: int,
        max_nodes: int,
        max_total_length: int,
        max_depth: int | None,
        change_of_variables_moves: bool,
        ac45_moves: bool,
        chunk_size: int,
        verbose: bool,
    ):
        self.max_n_gen = max_n_gen
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_total_length = max_total_length
        self.max_depth = max_depth
        self.change_of_variables_moves = change_of_variables_moves
        self.ac45_moves = ac45_moves
        self.chunk_size = chunk_size
        self.verbose = verbose

        self.initial_key, self.initial_flat = canonicalize_flat(
            initial_flat, n_gen, max_n_gen, max_length
        )
        self.params = MoveParams(
            n_gen=n_gen,
            max_n_gen=max_n_gen,
            max_length=max_length,
            max_steps_in_episode=max_depth or max_nodes,
        )
        self.apply_batch = build_apply_batch(
            self.params, change_of_variables_moves, ac45_moves
        )

    def solve(self) -> SearchResult:
        start_time = time.time()
        parents: dict[StateKey, tuple[StateKey | None, Action | None]] = {
            self.initial_key: (None, None)
        }
        flats: dict[StateKey, np.ndarray] = {self.initial_key: self.initial_flat}
        pq: list[tuple[int, int, int, StateKey]] = []
        seq = 0
        heapq.heappush(pq, (state_total_length(self.initial_key), 0, seq, self.initial_key))
        nodes_visited = 0
        max_priority_seen = state_total_length(self.initial_key)
        min_priority_seen = state_total_length(self.initial_key)

        while pq and nodes_visited < self.max_nodes:
            priority, depth, _, key = heapq.heappop(pq)
            nodes_visited += 1
            active_n_gen, _ = key

            if self.verbose:
                if priority > max_priority_seen:
                    print(
                        f"First state of priority {priority}, depth {depth}, "
                        f"n_gen {active_n_gen}, nodes {nodes_visited}",
                        flush=True,
                    )
                    max_priority_seen = priority
                if priority < min_priority_seen:
                    print(
                        f"First state of priority {priority}, depth {depth}, "
                        f"n_gen {active_n_gen}, nodes {nodes_visited}",
                        flush=True,
                    )
                    min_priority_seen = priority

            if is_trivial_key(key):
                elapsed = time.time() - start_time
                path_keys, path_actions = self._reconstruct(key, parents)
                return SearchResult(
                    solved=True,
                    nodes_visited=nodes_visited,
                    elapsed_sec=elapsed,
                    final_key=key,
                    path_keys=path_keys,
                    path_actions=path_actions,
                    seen_count=len(parents),
                )

            if self.max_depth is not None and depth >= self.max_depth:
                continue

            action_list = list(
                iter_actions(
                    key,
                    self.max_n_gen,
                    self.change_of_variables_moves,
                    self.ac45_moves,
                )
            )
            flat = flats[key]
            for action, next_flat, next_n_gen in apply_actions_chunked(
                self.apply_batch, flat, active_n_gen, action_list, self.chunk_size
            ):
                next_key, next_canon_flat = canonicalize_flat(
                    next_flat, next_n_gen, self.max_n_gen, self.max_length
                )
                if next_key == key or next_key in parents:
                    continue
                total_length = state_total_length(next_key)
                if total_length > self.max_total_length:
                    continue

                parents[next_key] = (key, action)
                flats[next_key] = next_canon_flat
                seq += 1
                heapq.heappush(pq, (total_length, depth + 1, seq, next_key))

        elapsed = time.time() - start_time
        return SearchResult(
            solved=False,
            nodes_visited=nodes_visited,
            elapsed_sec=elapsed,
            final_key=None,
            path_keys=[],
            path_actions=[],
            seen_count=len(parents),
        )

    @staticmethod
    def _reconstruct(
        final_key: StateKey,
        parents: dict[StateKey, tuple[StateKey | None, Action | None]],
    ) -> tuple[list[StateKey], list[Action]]:
        keys = []
        actions = []
        cur: StateKey | None = final_key
        while cur is not None:
            keys.append(cur)
            parent, action = parents[cur]
            if action is not None:
                actions.append(action)
            cur = parent
        keys.reverse()
        actions.reverse()
        return keys, actions


def load_dataset_presentation(
    dataset: str,
    idx: int,
    n_gen: int | None,
    max_n_gen: int | None,
    max_length: int,
) -> tuple[np.ndarray, int, int]:
    path_txt = os.path.join(REPO_ROOT, "data", f"{dataset}.txt")
    path_gz = f"{path_txt}.gz"
    if os.path.exists(path_txt):
        opener = open
        path = path_txt
    elif os.path.exists(path_gz):
        opener = gzip.open
        path = path_gz
    else:
        raise FileNotFoundError(f"could not find data/{dataset}.txt or .txt.gz")

    with opener(path, "rt") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx == idx:
                flat = np.asarray(ast.literal_eval(line.strip()), dtype=np.int8)
                break
        else:
            raise IndexError(f"dataset {dataset!r} has no row {idx}")

    if n_gen is None:
        if len(flat) % max_length != 0:
            raise ValueError(
                "--n_gen is required because row length is not divisible by --max_length"
            )
        n_gen = len(flat) // max_length
    if max_n_gen is None:
        max_n_gen = n_gen
    if n_gen > max_n_gen:
        raise ValueError("--n_gen must be <= --max_n_gen")
    if len(flat) != n_gen * max_length:
        flat = change_presentation_shape(flat, n_gen, max_length, max_n_gen=n_gen)
    if max_n_gen != n_gen:
        flat = change_presentation_shape(flat, n_gen, max_length, max_n_gen=max_n_gen)
    return np.asarray(flat, dtype=np.int8), n_gen, max_n_gen


def build_initial_from_relators(
    relator_texts: Sequence[str], n_gen: int | None, max_n_gen: int, max_length: int
) -> tuple[np.ndarray, int]:
    relators = [parse_word(text) for text in relator_texts]
    if n_gen is None:
        n_gen = max(len(relators), max((abs(v) for word in relators for v in word), default=0))
    if len(relators) != n_gen:
        raise ValueError(
            f"expected exactly {n_gen} relators for a balanced presentation, "
            f"got {len(relators)}"
        )
    if n_gen > max_n_gen:
        raise ValueError("--n_gen must be <= --max_n_gen")
    flat = convert_relator_list_to_presentation(
        relators, max_length, max_n_gen=max_n_gen
    )
    return np.asarray(flat, dtype=np.int8), n_gen


def format_state(key: StateKey) -> str:
    active_n_gen, relators = key
    relator_text = ", ".join(format_word(word) for word in relators)
    return f"n_gen={active_n_gen}: ({relator_text})"


def format_action(
    action: Action,
    max_n_gen: int,
    change_of_variables_moves: bool,
    ac45_moves: bool,
) -> str:
    branch, a1, a2, a3 = action
    s_branches = num_substitution_branches(max_n_gen)
    cov_branches = num_change_of_variables_branches(max_n_gen)
    if branch < s_branches:
        target, source = branch_to_pair(branch, max_n_gen)
        if max_n_gen == 2:
            return (
                f"S(update={branch}, base=0, source=1, inverse={a1}, "
                f"k_target={a2}, k_source={a3})"
            )
        return (
            f"S(target={target}, source={source}, inverse={a1}, "
            f"k_target={a2}, k_source={a3})"
        )

    if change_of_variables_moves and branch < s_branches + cov_branches:
        cov_branch = branch - s_branches
        remove_gen = cov_branch // max_n_gen
        iso_relator = cov_branch % max_n_gen
        return (
            f"COV(remove_gen={remove_gen}, iso_relator={iso_relator}, "
            f"z_inverse={a1}, z_start={a2}, z_len={a3 + 1})"
        )

    if ac45_moves and branch == add_generator_branch(max_n_gen, change_of_variables_moves):
        return "AC4(add_generator)"
    if ac45_moves and branch == delete_generator_branch(max_n_gen, change_of_variables_moves):
        return f"AC5(delete_gen={a1})"
    return f"unknown_action{action}"


def emit_result(
    result: SearchResult,
    max_n_gen: int,
    max_length: int,
    change_of_variables_moves: bool,
    ac45_moves: bool,
    show_path: bool,
    print_packed: bool,
    out_json: str | None,
) -> None:
    if result.solved:
        print(
            f"SOLVED depth={len(result.path_actions)} "
            f"nodes={result.nodes_visited} seen={result.seen_count} "
            f"time={result.elapsed_sec:.3f}s"
        )
    else:
        print(
            f"NOT SOLVED nodes={result.nodes_visited} seen={result.seen_count} "
            f"time={result.elapsed_sec:.3f}s"
        )

    if result.solved and show_path:
        for idx, key in enumerate(result.path_keys):
            print(f"Step {idx}: {format_state(key)}")
            if idx < len(result.path_actions):
                action = result.path_actions[idx]
                print(
                    "  "
                    + format_action(
                        action, max_n_gen, change_of_variables_moves, ac45_moves
                    )
                )

    if result.solved and print_packed:
        packed = [
            int(
                encode_action(
                    action,
                    max_length=max_length,
                    change_of_variables_moves=change_of_variables_moves,
                    ac45_moves=ac45_moves,
                    max_n_gen=max_n_gen,
                )
            )
            for action in result.path_actions
        ]
        print("packed_path =", packed)

    if out_json is not None:
        payload = {
            "solved": result.solved,
            "nodes_visited": result.nodes_visited,
            "seen_count": result.seen_count,
            "elapsed_sec": result.elapsed_sec,
            "path": [
                [[int(v) for v in word] for word in key[1]]
                for key in result.path_keys
            ],
            "actions": [list(map(int, action)) for action in result.path_actions],
        }
        with open(out_json, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="n-generator greedy best-first AC search"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--relators",
        nargs="+",
        help="balanced presentation relators, e.g. --relators xyxYXY xxxYYYY",
    )
    source.add_argument(
        "--dataset",
        help="dataset stem under data/; use with --idx",
    )
    parser.add_argument("--idx", type=int, default=0, help="dataset row index")
    parser.add_argument("--n_gen", type=int, default=None)
    parser.add_argument("--max_n_gen", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=24)
    parser.add_argument("--max_total_length", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=10000)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--change_of_variables_moves", action="store_true")
    parser.add_argument("--ac45_moves", action="store_true")
    parser.add_argument(
        "--stable_ac_moves",
        action="store_true",
        help="alias for enabling both --change_of_variables_moves and --ac45_moves",
    )
    parser.add_argument("--show_path", action="store_true")
    parser.add_argument("--print_packed", action="store_true")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    change_of_variables_moves = (
        args.change_of_variables_moves or args.stable_ac_moves
    )
    ac45_moves = args.ac45_moves or args.stable_ac_moves
    if args.max_nodes <= 0:
        raise SystemExit("--max_nodes must be positive")
    if args.max_length <= 0 or args.max_total_length <= 0:
        raise SystemExit("--max_length and --max_total_length must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive")

    if args.relators is not None:
        inferred_n = args.n_gen
        max_n_gen = args.max_n_gen or inferred_n or len(args.relators)
        initial_flat, n_gen = build_initial_from_relators(
            args.relators, inferred_n, max_n_gen, args.max_length
        )
    else:
        max_n_gen = args.max_n_gen or args.n_gen
        initial_flat, n_gen, max_n_gen = load_dataset_presentation(
            args.dataset, args.idx, args.n_gen, max_n_gen, args.max_length
        )

    if args.max_n_gen is None:
        max_n_gen = max(max_n_gen, n_gen)
    if n_gen > max_n_gen:
        raise SystemExit("--n_gen must be <= --max_n_gen")

    solver = GreedyNSolver(
        initial_flat=initial_flat,
        n_gen=n_gen,
        max_n_gen=max_n_gen,
        max_length=args.max_length,
        max_nodes=args.max_nodes,
        max_total_length=args.max_total_length,
        max_depth=args.max_depth,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(
            "Initial "
            + format_state(solver.initial_key)
            + f"; max_n_gen={max_n_gen}, max_length={args.max_length}, "
            + f"COV={change_of_variables_moves}, AC45={ac45_moves}",
            flush=True,
        )
    result = solver.solve()
    emit_result(
        result,
        max_n_gen=max_n_gen,
        max_length=args.max_length,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
        show_path=args.show_path,
        print_packed=args.print_packed,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
