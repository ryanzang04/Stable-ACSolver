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
    python scripts/greedy_search_n.py --dataset 1190MS --all --max_nodes 10000 \
        --out_jsonl greedy_1190MS.jsonl
    python scripts/greedy_search_n.py --dataset 1190MS --all --max_nodes 10000 \
        --wandb --wandb_project ACSolverX-greedy
    python scripts/greedy_search_n.py --dataset 1190MS --compare_cov --all \
        --max_nodes 10000 --out_jsonl cov_compare.jsonl
    python scripts/greedy_search_n.py --dataset AC19 --start 0 --end 100 \
        --max_nodes 5000
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
from collections import deque
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


# Top 10 entries from the existing all_z_word.csv COV-path analysis.
# Integer words use x=1, y=2 and uppercase letters for inverses.
W_COV: tuple[tuple[int, ...], ...] = (
    (-2,),                 # Y
    (-1,),                 # X
    (1,),                  # x
    (-1, 2),               # Xy
    (2, 1),                # yx
    (-2, -2, -2, -1),     # YYYX
    (-2, -1),              # YX
    (-2, -1, 2, 1),        # YXyx
    (1, -2),               # xY
    (-2, -2, -1),          # YYX
)


@dataclass(frozen=True)
class SeededCovAction:
    word: tuple[int, ...]
    sources: tuple[str, ...]


PathAction = Action | SeededCovAction


@dataclass(frozen=True)
class SeedWord:
    word: tuple[int, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SeededPresentation:
    action: SeededCovAction
    key: StateKey
    flat: np.ndarray


@dataclass(frozen=True)
class SeededSearchFrontier:
    seeds: list[SeededPresentation]
    trivial_stabilization_key: StateKey
    candidate_count: int
    duplicate_count: int


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    nodes_visited: int
    elapsed_sec: float
    final_key: StateKey | None
    path_keys: list[StateKey]
    path_actions: list[PathAction]
    seen_count: int
    seeded_cov_enabled: bool = False
    seeded_cov_seed_count: int = 0
    seeded_cov_frontier_size: int = 0
    seeded_cov_duplicate_count: int = 0
    seeded_cov_trivial_stabilization: StateKey | None = None


@dataclass(frozen=True)
class MoveParams:
    n_gen: int = 2
    max_n_gen: int = 2
    max_length: int = 24
    max_steps_in_episode: int = 200


@dataclass(frozen=True)
class DatasetRow:
    idx: int
    flat: np.ndarray
    n_gen: int
    max_n_gen: int


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


def seeded_cov_word_candidates(
    original_relators: Sequence[Sequence[int]],
) -> list[SeedWord]:
    if len(original_relators) != 2:
        raise ValueError("--seeded_cov_moves requires an initial rank-two presentation")

    ordered: dict[tuple[int, ...], list[str]] = {}

    def add(word: Sequence[int], source: str) -> None:
        normalized = tuple(int(code) for code in word if int(code) != 0)
        sources = ordered.setdefault(normalized, [])
        if source not in sources:
            sources.append(source)

    add(inverse_word(original_relators[0]), "r1^-1")
    add(inverse_word(original_relators[1]), "r2^-1")
    for word in W_COV:
        add(word, "W_COV")
    return [
        SeedWord(word=word, sources=tuple(sources))
        for word, sources in ordered.items()
    ]


def seeded_definition_word(word: Sequence[int], new_generator: int = 3) -> tuple[int, ...]:
    """Return z w, intentionally defining z = w^-1."""
    return (int(new_generator),) + tuple(int(code) for code in word)


def build_seeded_presentation(
    original_relators: Sequence[Sequence[int]],
    seed_word: SeedWord,
    max_n_gen: int,
    max_length: int,
) -> SeededPresentation | None:
    if len(original_relators) != 2:
        raise ValueError("--seeded_cov_moves requires an initial rank-two presentation")
    seeded_n_gen = 3
    if max_n_gen < seeded_n_gen:
        raise ValueError("seeded COV requires max_n_gen >= 3")

    definition = seeded_definition_word(seed_word.word, seeded_n_gen)
    raw_relators = [tuple(word) for word in original_relators] + [definition]
    if any(len(relator) > max_length for relator in raw_relators):
        return None

    key: StateKey = (
        seeded_n_gen,
        canonicalize_relators(raw_relators, seeded_n_gen),
    )
    return SeededPresentation(
        action=SeededCovAction(seed_word.word, seed_word.sources),
        key=key,
        flat=key_to_flat(key, max_n_gen, max_length),
    )


def build_trivial_stabilization(
    original_relators: Sequence[Sequence[int]],
    max_n_gen: int,
    max_length: int,
) -> StateKey:
    """Return canonical <x,y,z | r1,r2,z>."""
    if len(original_relators) != 2:
        raise ValueError("trivial stabilization requires a rank-two presentation")
    if max_n_gen < 3:
        raise ValueError("trivial stabilization requires max_n_gen >= 3")
    return (
        3,
        canonicalize_relators(
            [*original_relators, (3,)],
            active_n_gen=3,
        ),
    )


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


def iter_change_of_variables_actions(
    key: StateKey,
    max_n_gen: int,
    cov_window_min_length: int = 1,
    cov_window_max_length: int | None = None,
    cov_window_relator_margin: int = 1,
) -> Iterator[Action]:
    active_n_gen, relators = key
    for remove_gen in range(active_n_gen):
        old_code = remove_gen + 1
        for iso_relator, word in enumerate(relators):
            rel_len = len(word)
            if rel_len <= 1:
                continue
            branch = change_of_variables_branch(remove_gen, iso_relator, max_n_gen)
            max_z_len = rel_len - cov_window_relator_margin
            if cov_window_max_length is not None:
                max_z_len = min(max_z_len, cov_window_max_length)
            for z_start in range(rel_len):
                for z_len in range(cov_window_min_length, max_z_len + 1):
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
    cov_window_min_length: int = 1,
    cov_window_max_length: int | None = None,
    cov_window_relator_margin: int = 1,
) -> Iterator[Action]:
    yield from iter_substitution_actions(key, max_n_gen)
    if change_of_variables_moves:
        yield from iter_change_of_variables_actions(
            key,
            max_n_gen,
            cov_window_min_length,
            cov_window_max_length,
            cov_window_relator_margin,
        )
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


def build_seeded_search_frontier(
    initial_key: StateKey,
    original_relators: Sequence[Sequence[int]],
    max_n_gen: int,
    max_length: int,
    max_total_length: int,
) -> SeededSearchFrontier:
    """Build unique root seed states while reserving the trivial stabilization."""
    trivial_stabilization_key = build_trivial_stabilization(
        original_relators,
        max_n_gen,
        max_length,
    )
    blocked_or_seen = {initial_key, trivial_stabilization_key}
    seeds: list[SeededPresentation] = []
    candidate_count = 0
    duplicate_count = 0

    for seed_word in seeded_cov_word_candidates(original_relators):
        seeded = build_seeded_presentation(
            original_relators,
            seed_word,
            max_n_gen,
            max_length,
        )
        if seeded is None:
            continue
        candidate_count += 1
        if state_total_length(seeded.key) > max_total_length:
            continue
        if seeded.key in blocked_or_seen:
            duplicate_count += 1
            continue
        blocked_or_seen.add(seeded.key)
        seeds.append(seeded)

    return SeededSearchFrontier(
        seeds=seeds,
        trivial_stabilization_key=trivial_stabilization_key,
        candidate_count=candidate_count,
        duplicate_count=duplicate_count,
    )


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
        apply_batch=None,
        seeded_cov_moves: bool = False,
        cov_window_min_length: int = 1,
        cov_window_max_length: int | None = None,
        cov_window_relator_margin: int = 1,
    ):
        self.max_n_gen = max_n_gen
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_total_length = max_total_length
        self.max_depth = max_depth
        self.change_of_variables_moves = change_of_variables_moves
        self.ac45_moves = ac45_moves
        self.seeded_cov_moves = seeded_cov_moves
        self.cov_window_min_length = cov_window_min_length
        self.cov_window_max_length = cov_window_max_length
        self.cov_window_relator_margin = cov_window_relator_margin
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.original_relators = tuple(
            flat_to_relators(initial_flat, n_gen, max_length)
        )
        self.seeded_frontier: SeededSearchFrontier | None = None

        if seeded_cov_moves and n_gen != 2:
            raise ValueError("--seeded_cov_moves requires an initial rank-two presentation")
        if seeded_cov_moves and max_n_gen < n_gen + 1:
            raise ValueError("--seeded_cov_moves requires max_n_gen >= n_gen + 1")
        if seeded_cov_moves and not change_of_variables_moves:
            raise ValueError("seeded COV requires ordinary COV support")

        self.initial_key, self.initial_flat = canonicalize_flat(
            initial_flat, n_gen, max_n_gen, max_length
        )
        self.params = MoveParams(
            n_gen=n_gen,
            max_n_gen=max_n_gen,
            max_length=max_length,
            max_steps_in_episode=max_depth or max_nodes,
        )
        self.apply_batch = apply_batch or build_apply_batch(
            self.params, change_of_variables_moves, ac45_moves
        )

    def solve(self) -> SearchResult:
        start_time = time.time()
        parents: dict[StateKey, tuple[StateKey | None, PathAction | None]] = {
            self.initial_key: (None, None)
        }
        flats: dict[StateKey, np.ndarray] = {self.initial_key: self.initial_flat}
        pq: list[tuple[int, int, int, StateKey]] = []
        seq = 0
        if self.seeded_cov_moves:
            self.seeded_frontier = build_seeded_search_frontier(
                initial_key=self.initial_key,
                original_relators=self.original_relators,
                max_n_gen=self.max_n_gen,
                max_length=self.max_length,
                max_total_length=self.max_total_length,
            )
            parents[self.seeded_frontier.trivial_stabilization_key] = (None, None)
            for seed in self.seeded_frontier.seeds:
                parents[seed.key] = (self.initial_key, seed.action)
                flats[seed.key] = seed.flat
                seq += 1
                heapq.heappush(
                    pq,
                    (state_total_length(seed.key), 1, seq, seed.key),
                )
        else:
            heapq.heappush(
                pq,
                (state_total_length(self.initial_key), 0, seq, self.initial_key),
            )
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
                    seeded_cov_enabled=self.seeded_cov_moves,
                    seeded_cov_seed_count=(
                        self.seeded_frontier.candidate_count
                        if self.seeded_frontier is not None else 0
                    ),
                    seeded_cov_frontier_size=(
                        len(self.seeded_frontier.seeds)
                        if self.seeded_frontier is not None else 0
                    ),
                    seeded_cov_duplicate_count=(
                        self.seeded_frontier.duplicate_count
                        if self.seeded_frontier is not None else 0
                    ),
                    seeded_cov_trivial_stabilization=(
                        self.seeded_frontier.trivial_stabilization_key
                        if self.seeded_frontier is not None else None
                    ),
                )

            if self.max_depth is not None and depth >= self.max_depth:
                continue

            action_list = list(
                iter_actions(
                    key,
                    self.max_n_gen,
                    self.change_of_variables_moves,
                    self.ac45_moves,
                    self.cov_window_min_length,
                    self.cov_window_max_length,
                    self.cov_window_relator_margin,
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
            seeded_cov_enabled=self.seeded_cov_moves,
            seeded_cov_seed_count=(
                self.seeded_frontier.candidate_count
                if self.seeded_frontier is not None else 0
            ),
            seeded_cov_frontier_size=(
                len(self.seeded_frontier.seeds)
                if self.seeded_frontier is not None else 0
            ),
            seeded_cov_duplicate_count=(
                self.seeded_frontier.duplicate_count
                if self.seeded_frontier is not None else 0
            ),
            seeded_cov_trivial_stabilization=(
                self.seeded_frontier.trivial_stabilization_key
                if self.seeded_frontier is not None else None
            ),
        )

    @staticmethod
    def _reconstruct(
        final_key: StateKey,
        parents: dict[StateKey, tuple[StateKey | None, PathAction | None]],
    ) -> tuple[list[StateKey], list[PathAction]]:
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


def dataset_path(dataset: str) -> str:
    path_txt = os.path.join(REPO_ROOT, "data", f"{dataset}.txt")
    path_gz = f"{path_txt}.gz"
    if os.path.exists(path_txt):
        return path_txt
    if os.path.exists(path_gz):
        return path_gz
    raise FileNotFoundError(f"could not find data/{dataset}.txt or .txt.gz")


def open_dataset(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def normalize_dataset_flat(
    flat: Sequence[int],
    n_gen: int | None,
    max_n_gen: int | None,
    max_length: int,
) -> tuple[np.ndarray, int, int]:
    flat = np.asarray(flat, dtype=np.int8)
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


def load_dataset_presentation(
    dataset: str,
    idx: int,
    n_gen: int | None,
    max_n_gen: int | None,
    max_length: int,
) -> tuple[np.ndarray, int, int]:
    path = dataset_path(dataset)
    with open_dataset(path) as handle:
        for line_idx, line in enumerate(handle):
            if line_idx == idx:
                flat = ast.literal_eval(line.strip())
                return normalize_dataset_flat(flat, n_gen, max_n_gen, max_length)
    raise IndexError(f"dataset {dataset!r} has no row {idx}")


def iter_dataset_rows(
    dataset: str,
    start: int,
    end: int | None,
    n_gen: int | None,
    max_n_gen: int | None,
    max_length: int,
) -> Iterator[DatasetRow]:
    path = dataset_path(dataset)
    with open_dataset(path) as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start:
                continue
            if end is not None and line_idx >= end:
                break
            flat, row_n_gen, row_max_n_gen = normalize_dataset_flat(
                ast.literal_eval(line.strip()), n_gen, max_n_gen, max_length
            )
            yield DatasetRow(
                idx=line_idx,
                flat=flat,
                n_gen=row_n_gen,
                max_n_gen=row_max_n_gen,
            )


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
    action: PathAction,
    max_n_gen: int,
    change_of_variables_moves: bool,
    ac45_moves: bool,
) -> str:
    if isinstance(action, SeededCovAction):
        return (
            f"SEEDED_COV_ADD(w={format_word(action.word)}, "
            f"sources={','.join(action.sources)})"
        )
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


def action_payload(action: PathAction):
    if isinstance(action, SeededCovAction):
        return {
            "type": "seeded_cov_add",
            "word": [int(code) for code in action.word],
            "word_text": format_word(action.word),
            "sources": list(action.sources),
        }
    return [int(value) for value in action]


def presentation_payload(key: StateKey) -> dict:
    return {
        "n_gen": int(key[0]),
        "relators": [
            [int(code) for code in relator]
            for relator in key[1]
        ],
        "relators_text": [format_word(relator) for relator in key[1]],
    }


def seeded_cov_solution_metadata(result: SearchResult) -> dict | None:
    if not result.solved or not result.path_actions:
        return None
    seed_action = result.path_actions[0]
    if not isinstance(seed_action, SeededCovAction):
        return None
    if len(result.path_actions) < 2 or len(result.path_keys) < 3:
        raise AssertionError("seeded search path is missing its first search step")
    first_search_action = result.path_actions[1]
    if isinstance(first_search_action, SeededCovAction):
        raise AssertionError("seeded-variable addition was applied more than once")
    return {
        "word": [int(code) for code in seed_action.word],
        "word_text": format_word(seed_action.word),
        "sources": list(seed_action.sources),
        "seeded_presentation": presentation_payload(result.path_keys[1]),
        "first_search_action": [int(value) for value in first_search_action],
        "post_first_action_presentation": presentation_payload(
            result.path_keys[2]
        ),
    }


def result_payload(result: SearchResult, idx: int | None = None) -> dict:
    payload = {
        "solved": result.solved,
        "nodes_visited": result.nodes_visited,
        "seen_count": result.seen_count,
        "elapsed_sec": result.elapsed_sec,
        "path_length": len(result.path_actions) if result.solved else -1,
        "path": [
            [[int(v) for v in word] for word in key[1]]
            for key in result.path_keys
        ],
        "actions": [action_payload(action) for action in result.path_actions],
    }
    if result.seeded_cov_enabled:
        payload["seeded_cov"] = {
            "seed_count": result.seeded_cov_seed_count,
            "initial_frontier_size": result.seeded_cov_frontier_size,
            "duplicates_removed": result.seeded_cov_duplicate_count,
            "previsited_trivial_stabilization": (
                presentation_payload(result.seeded_cov_trivial_stabilization)
                if result.seeded_cov_trivial_stabilization is not None else None
            ),
            "solution": seeded_cov_solution_metadata(result),
        }
    if idx is not None:
        payload["idx"] = idx
    return payload


def init_wandb_run(
    args: argparse.Namespace,
    change_of_variables_moves: bool,
    ac45_moves: bool,
    batch_mode: bool,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Install it or run without --wandb."
        ) from exc

    config = {
        "dataset": args.dataset,
        "idx": args.idx,
        "all": args.all,
        "start": args.start,
        "end": args.end,
        "n_gen": args.n_gen,
        "max_n_gen": args.max_n_gen,
        "max_length": args.max_length,
        "max_total_length": args.max_total_length,
        "max_nodes": args.max_nodes,
        "max_depth": args.max_depth,
        "chunk_size": args.chunk_size,
        "cov_window_min_length": args.cov_window_min_length,
        "cov_window_max_length": args.cov_window_max_length,
        "cov_window_relator_margin": args.cov_window_relator_margin,
        "change_of_variables_moves": change_of_variables_moves,
        "seeded_cov_moves": args.seeded_cov_moves,
        "ac45_moves": ac45_moves,
        "stable_ac_moves": args.stable_ac_moves,
        "compare_cov": args.compare_cov,
        "compare_cov_paths": args.compare_cov_paths,
        "batch_mode": batch_mode,
        "wandb_window": args.wandb_window,
        "wandb_log_every": args.wandb_log_every,
        "index_summary_limit": args.index_summary_limit,
        "wandb_table": args.wandb_table,
    }
    name = args.wandb_run_name or None
    tags = args.wandb_tags if args.wandb_tags else None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=name,
        tags=tags,
        config=config,
        mode=args.wandb_mode,
    )


def log_single_wandb(wandb_run, result: SearchResult, idx: int | None = None) -> None:
    if wandb_run is None:
        return
    solved = int(result.solved)
    wandb_run.log(
        {
            "idx": -1 if idx is None else idx,
            "solved": solved,
            "path_length": len(result.path_actions) if result.solved else -1,
            "nodes_visited": result.nodes_visited,
            "seen_count": result.seen_count,
            "elapsed_sec": result.elapsed_sec,
            "solve_rate": float(solved),
            "num_solved": solved,
            "processed": 1,
        },
        step=1,
    )
    wandb_run.summary["solve_rate"] = float(solved)
    wandb_run.summary["num_solved"] = solved
    wandb_run.summary["processed"] = 1


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
        with open(out_json, "w") as handle:
            json.dump(result_payload(result), handle, indent=2)
            handle.write("\n")


def make_solver(
    initial_flat: np.ndarray,
    n_gen: int,
    max_n_gen: int,
    args: argparse.Namespace,
    change_of_variables_moves: bool,
    ac45_moves: bool,
    verbose: bool,
    apply_batch=None,
) -> GreedyNSolver:
    return GreedyNSolver(
        initial_flat=initial_flat,
        n_gen=n_gen,
        max_n_gen=max_n_gen,
        max_length=args.max_length,
        max_nodes=args.max_nodes,
        max_total_length=args.max_total_length,
        max_depth=args.max_depth,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
        seeded_cov_moves=args.seeded_cov_moves,
        cov_window_min_length=args.cov_window_min_length,
        cov_window_max_length=args.cov_window_max_length,
        cov_window_relator_margin=args.cov_window_relator_margin,
        chunk_size=args.chunk_size,
        verbose=verbose,
        apply_batch=apply_batch,
    )


def solve_dataset_row(
    row: DatasetRow,
    args: argparse.Namespace,
    change_of_variables_moves: bool,
    ac45_moves: bool,
    apply_batch_cache: dict,
    verbose: bool = False,
) -> SearchResult:
    cache_key = (
        row.max_n_gen,
        args.max_length,
        bool(change_of_variables_moves),
        bool(ac45_moves),
    )
    if cache_key not in apply_batch_cache:
        params = MoveParams(
            n_gen=row.n_gen,
            max_n_gen=row.max_n_gen,
            max_length=args.max_length,
            max_steps_in_episode=args.max_depth or args.max_nodes,
        )
        apply_batch_cache[cache_key] = build_apply_batch(
            params, change_of_variables_moves, ac45_moves
        )

    solver = make_solver(
        row.flat,
        row.n_gen,
        row.max_n_gen,
        args,
        change_of_variables_moves,
        ac45_moves,
        verbose=verbose,
        apply_batch=apply_batch_cache[cache_key],
    )
    return solver.solve()


def capped_indices(indices: Sequence[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    return [int(idx) for idx in indices[:limit]]


def add_index_summaries(wandb_run, prefix: str, indices: Sequence[int],
                        limit: int) -> None:
    if wandb_run is None:
        return
    key_prefix = f"{prefix}_" if prefix else ""
    wandb_run.summary[f"{key_prefix}indices_count"] = len(indices)
    wandb_run.summary[f"{key_prefix}indices"] = capped_indices(indices, limit)
    wandb_run.summary[f"{key_prefix}indices_truncated"] = len(indices) > limit


def maybe_log_wandb_table(wandb_run, key: str, columns: list[str],
                          rows: list[list], enabled: bool) -> None:
    if wandb_run is None or not enabled:
        return
    import wandb
    wandb_run.log({key: wandb.Table(columns=columns, data=rows)})


def run_dataset_batch(
    args: argparse.Namespace,
    change_of_variables_moves: bool,
    ac45_moves: bool,
) -> None:
    start = args.start if args.start is not None else 0
    end = args.end
    if start < 0:
        raise SystemExit("--start must be non-negative")
    if end is not None and end <= start:
        raise SystemExit("--end must be greater than --start")
    if args.show_path:
        raise SystemExit("--show_path is only supported for a single presentation")
    if args.print_packed:
        raise SystemExit("--print_packed is only supported for a single presentation")

    out_jsonl = args.out_jsonl or args.out_json
    out_handle = open(out_jsonl, "w") if out_jsonl is not None else None
    wandb_run = init_wandb_run(
        args, change_of_variables_moves, ac45_moves, batch_mode=True
    )
    solved_window = deque(maxlen=args.wandb_window)
    apply_batch_cache = {}
    total = 0
    solved = 0
    total_nodes = 0
    solved_indices: list[int] = []
    table_rows: list[list] = []
    t0 = time.time()

    try:
        for row in iter_dataset_rows(
            args.dataset,
            start,
            end,
            args.n_gen,
            args.max_n_gen,
            args.max_length,
        ):
            result = solve_dataset_row(
                row,
                args,
                change_of_variables_moves,
                ac45_moves,
                apply_batch_cache,
            )
            total += 1
            row_solved = int(result.solved)
            solved += row_solved
            if result.solved:
                solved_indices.append(row.idx)
            solved_window.append(row_solved)
            total_nodes += result.nodes_visited
            solve_rate = solved / total
            rolling_solve_rate = sum(solved_window) / len(solved_window)
            depth = len(result.path_actions) if result.solved else -1
            solved_idx = row.idx if result.solved else -1
            table_rows.append(
                [
                    row.idx,
                    bool(result.solved),
                    depth,
                    result.nodes_visited,
                    result.seen_count,
                    result.elapsed_sec,
                    solve_rate,
                ]
            )

            if not args.quiet:
                status = "SOLVED" if result.solved else "not solved"
                print(
                    f"[{row.idx}] {status} depth={depth} "
                    f"nodes={result.nodes_visited} seen={result.seen_count} "
                    f"time={result.elapsed_sec:.3f}s solve_rate={solve_rate:.3f}",
                    flush=True,
                )

            if out_handle is not None:
                payload = result_payload(result, idx=row.idx)
                payload.update(
                    {
                        "processed": total,
                        "num_solved": solved,
                        "solved_idx": solved_idx,
                        "solve_rate": solve_rate,
                        "rolling_solve_rate": rolling_solve_rate,
                        "total_nodes": total_nodes,
                    }
                )
                json.dump(payload, out_handle)
                out_handle.write("\n")
                out_handle.flush()

            if wandb_run is not None and total % args.wandb_log_every == 0:
                elapsed_so_far = time.time() - t0
                wandb_run.log(
                    {
                        "idx": row.idx,
                        "processed": total,
                        "solved": row_solved,
                        "solved_idx": solved_idx,
                        "num_solved": solved,
                        "solve_rate": solve_rate,
                        "rolling_solve_rate": rolling_solve_rate,
                        "path_length": depth,
                        "nodes_visited": result.nodes_visited,
                        "seen_count": result.seen_count,
                        "elapsed_sec": result.elapsed_sec,
                        "total_nodes": total_nodes,
                        "rows_per_second": total / max(elapsed_so_far, 1e-9),
                        "active_n_gen": row.n_gen,
                        "max_n_gen": row.max_n_gen,
                    },
                    step=total,
                )
    finally:
        if out_handle is not None:
            out_handle.close()

    elapsed = time.time() - t0
    final_solve_rate = solved / total if total else 0.0
    if wandb_run is not None:
        wandb_run.summary["processed"] = total
        wandb_run.summary["num_solved"] = solved
        wandb_run.summary["num_unsolved"] = total - solved
        wandb_run.summary["solve_rate"] = final_solve_rate
        wandb_run.summary["total_nodes"] = total_nodes
        wandb_run.summary["elapsed_sec"] = elapsed
        add_index_summaries(
            wandb_run, "solved", solved_indices, args.index_summary_limit
        )
        maybe_log_wandb_table(
            wandb_run,
            "results",
            [
                "idx",
                "solved",
                "path_length",
                "nodes_visited",
                "seen_count",
                "elapsed_sec",
                "solve_rate",
            ],
            table_rows,
            args.wandb_table,
        )
        wandb_run.finish()
    print(
        f"SUMMARY rows={total} solved={solved} unsolved={total - solved} "
        f"solve_rate={final_solve_rate:.3f} nodes={total_nodes} time={elapsed:.3f}s"
    )


def comparison_result_payload(
    row: DatasetRow,
    without_cov: SearchResult,
    with_cov: SearchResult,
    processed: int,
    counts: dict[str, int],
    include_paths: bool = False,
) -> dict:
    without_depth = len(without_cov.path_actions) if without_cov.solved else -1
    with_depth = len(with_cov.path_actions) if with_cov.solved else -1
    total = max(processed, 1)
    payload = {
        "idx": row.idx,
        "processed": processed,
        "without_cov_solved": bool(without_cov.solved),
        "with_cov_solved": bool(with_cov.solved),
        "cov_only_solved": bool(with_cov.solved and not without_cov.solved),
        "without_cov_only_solved": bool(without_cov.solved and not with_cov.solved),
        "both_solved": bool(without_cov.solved and with_cov.solved),
        "neither_solved": bool(not without_cov.solved and not with_cov.solved),
        "without_cov_path_length": without_depth,
        "with_cov_path_length": with_depth,
        "path_length_delta_cov_minus_without": (
            with_depth - without_depth
            if without_cov.solved and with_cov.solved else None
        ),
        "without_cov_nodes": without_cov.nodes_visited,
        "with_cov_nodes": with_cov.nodes_visited,
        "without_cov_seen": without_cov.seen_count,
        "with_cov_seen": with_cov.seen_count,
        "without_cov_elapsed_sec": without_cov.elapsed_sec,
        "with_cov_elapsed_sec": with_cov.elapsed_sec,
        "without_cov_solve_rate": counts["without_cov_solved"] / total,
        "with_cov_solve_rate": counts["with_cov_solved"] / total,
        "cov_only_count": counts["cov_only"],
        "without_cov_only_count": counts["without_cov_only"],
        "both_solved_count": counts["both"],
        "neither_solved_count": counts["neither"],
    }
    if include_paths:
        payload["without_cov_result"] = result_payload(without_cov, row.idx)
        payload["with_cov_result"] = result_payload(with_cov, row.idx)
    return payload


def run_cov_comparison(args: argparse.Namespace, ac45_moves: bool) -> None:
    start = args.start
    if start is None:
        start = 0 if args.all or args.end is not None else args.idx
    end = args.end
    if end is None and not args.all:
        end = start + 1
    if start < 0:
        raise SystemExit("--start/--idx must be non-negative")
    if end is not None and end <= start:
        raise SystemExit("--end must be greater than the first index")
    if args.show_path:
        raise SystemExit("--show_path is not supported in --compare_cov mode")
    if args.print_packed:
        raise SystemExit("--print_packed is not supported in --compare_cov mode")

    out_jsonl = args.out_jsonl or args.out_json
    out_handle = open(out_jsonl, "w") if out_jsonl is not None else None
    wandb_run = init_wandb_run(
        args, change_of_variables_moves=True, ac45_moves=ac45_moves,
        batch_mode=True,
    )
    apply_batch_cache = {}
    counts = {
        "without_cov_solved": 0,
        "with_cov_solved": 0,
        "cov_only": 0,
        "without_cov_only": 0,
        "both": 0,
        "neither": 0,
    }
    without_cov_solved_indices: list[int] = []
    with_cov_solved_indices: list[int] = []
    cov_only_indices: list[int] = []
    without_cov_only_indices: list[int] = []
    both_solved_indices: list[int] = []
    table_rows: list[list] = []
    total_nodes = 0
    total = 0
    t0 = time.time()

    try:
        for row in iter_dataset_rows(
            args.dataset,
            start,
            end,
            args.n_gen,
            args.max_n_gen,
            args.max_length,
        ):
            without_cov = solve_dataset_row(
                row, args, False, ac45_moves, apply_batch_cache
            )
            with_cov = solve_dataset_row(
                row, args, True, ac45_moves, apply_batch_cache
            )
            total += 1
            no_solved = int(without_cov.solved)
            cov_solved = int(with_cov.solved)
            cov_only = bool(with_cov.solved and not without_cov.solved)
            without_only = bool(without_cov.solved and not with_cov.solved)
            both = bool(without_cov.solved and with_cov.solved)
            neither = bool(not without_cov.solved and not with_cov.solved)
            counts["without_cov_solved"] += no_solved
            counts["with_cov_solved"] += cov_solved
            counts["cov_only"] += int(cov_only)
            counts["without_cov_only"] += int(without_only)
            counts["both"] += int(both)
            counts["neither"] += int(neither)
            total_nodes += without_cov.nodes_visited + with_cov.nodes_visited

            if without_cov.solved:
                without_cov_solved_indices.append(row.idx)
            if with_cov.solved:
                with_cov_solved_indices.append(row.idx)
            if cov_only:
                cov_only_indices.append(row.idx)
            if without_only:
                without_cov_only_indices.append(row.idx)
            if both:
                both_solved_indices.append(row.idx)

            payload = comparison_result_payload(
                row,
                without_cov,
                with_cov,
                total,
                counts,
                include_paths=args.compare_cov_paths,
            )
            without_depth = payload["without_cov_path_length"]
            with_depth = payload["with_cov_path_length"]
            relation = (
                "cov_only" if cov_only else
                "without_cov_only" if without_only else
                "both" if both else
                "neither"
            )
            table_rows.append(
                [
                    row.idx,
                    bool(without_cov.solved),
                    bool(with_cov.solved),
                    relation,
                    without_depth,
                    with_depth,
                    without_cov.nodes_visited,
                    with_cov.nodes_visited,
                    without_cov.elapsed_sec,
                    with_cov.elapsed_sec,
                ]
            )

            if not args.quiet:
                print(
                    f"[{row.idx}] no_cov={bool(without_cov.solved)} "
                    f"cov={bool(with_cov.solved)} relation={relation} "
                    f"no_depth={without_depth} cov_depth={with_depth} "
                    f"cov_only={counts['cov_only']} "
                    f"cov_rate={counts['with_cov_solved'] / total:.3f} "
                    f"no_cov_rate={counts['without_cov_solved'] / total:.3f}",
                    flush=True,
                )

            if out_handle is not None:
                json.dump(payload, out_handle)
                out_handle.write("\n")
                out_handle.flush()

            if wandb_run is not None and total % args.wandb_log_every == 0:
                elapsed_so_far = time.time() - t0
                wandb_run.log(
                    {
                        "idx": row.idx,
                        "processed": total,
                        "without_cov/solved": no_solved,
                        "with_cov/solved": cov_solved,
                        "without_cov/solve_rate": counts["without_cov_solved"] / total,
                        "with_cov/solve_rate": counts["with_cov_solved"] / total,
                        "cov_only_solved": int(cov_only),
                        "without_cov_only_solved": int(without_only),
                        "both_solved": int(both),
                        "neither_solved": int(neither),
                        "cov_only_count": counts["cov_only"],
                        "without_cov_only_count": counts["without_cov_only"],
                        "cov_only_idx": row.idx if cov_only else -1,
                        "without_cov_only_idx": row.idx if without_only else -1,
                        "without_cov/path_length": without_depth,
                        "with_cov/path_length": with_depth,
                        "without_cov/nodes_visited": without_cov.nodes_visited,
                        "with_cov/nodes_visited": with_cov.nodes_visited,
                        "without_cov/elapsed_sec": without_cov.elapsed_sec,
                        "with_cov/elapsed_sec": with_cov.elapsed_sec,
                        "total_nodes": total_nodes,
                        "rows_per_second": total / max(elapsed_so_far, 1e-9),
                    },
                    step=total,
                )
    finally:
        if out_handle is not None:
            out_handle.close()

    elapsed = time.time() - t0
    without_rate = counts["without_cov_solved"] / total if total else 0.0
    with_rate = counts["with_cov_solved"] / total if total else 0.0
    if wandb_run is not None:
        wandb_run.summary["processed"] = total
        wandb_run.summary["without_cov_num_solved"] = counts["without_cov_solved"]
        wandb_run.summary["with_cov_num_solved"] = counts["with_cov_solved"]
        wandb_run.summary["without_cov_solve_rate"] = without_rate
        wandb_run.summary["with_cov_solve_rate"] = with_rate
        wandb_run.summary["cov_only_count"] = counts["cov_only"]
        wandb_run.summary["without_cov_only_count"] = counts["without_cov_only"]
        wandb_run.summary["both_solved_count"] = counts["both"]
        wandb_run.summary["neither_solved_count"] = counts["neither"]
        wandb_run.summary["total_nodes"] = total_nodes
        wandb_run.summary["elapsed_sec"] = elapsed
        add_index_summaries(
            wandb_run, "without_cov_solved", without_cov_solved_indices,
            args.index_summary_limit,
        )
        add_index_summaries(
            wandb_run, "with_cov_solved", with_cov_solved_indices,
            args.index_summary_limit,
        )
        add_index_summaries(
            wandb_run, "cov_only", cov_only_indices, args.index_summary_limit
        )
        add_index_summaries(
            wandb_run, "without_cov_only", without_cov_only_indices,
            args.index_summary_limit,
        )
        add_index_summaries(
            wandb_run, "both_solved", both_solved_indices,
            args.index_summary_limit,
        )
        maybe_log_wandb_table(
            wandb_run,
            "cov_comparison",
            [
                "idx",
                "without_cov_solved",
                "with_cov_solved",
                "relation",
                "without_cov_path_length",
                "with_cov_path_length",
                "without_cov_nodes",
                "with_cov_nodes",
                "without_cov_elapsed_sec",
                "with_cov_elapsed_sec",
            ],
            table_rows,
            args.wandb_table,
        )
        wandb_run.finish()

    print(
        "SUMMARY "
        f"rows={total} "
        f"without_cov_solved={counts['without_cov_solved']} "
        f"with_cov_solved={counts['with_cov_solved']} "
        f"without_cov_rate={without_rate:.3f} "
        f"with_cov_rate={with_rate:.3f} "
        f"cov_only={counts['cov_only']} "
        f"without_cov_only={counts['without_cov_only']} "
        f"both={counts['both']} neither={counts['neither']} "
        f"nodes={total_nodes} time={elapsed:.3f}s"
    )
    if cov_only_indices:
        print("COV_ONLY_INDICES", cov_only_indices)
    if without_cov_only_indices:
        print("WITHOUT_COV_ONLY_INDICES", without_cov_only_indices)


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
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every row in --dataset, starting at --start if provided",
    )
    parser.add_argument("--start", type=int, default=None, help="first dataset index")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="exclusive end dataset index; omit with --all to run to EOF",
    )
    parser.add_argument("--n_gen", type=int, default=None)
    parser.add_argument("--max_n_gen", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=24)
    parser.add_argument("--max_total_length", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=10000)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--change_of_variables_moves", action="store_true")
    parser.add_argument(
        "--cov_window_min_length",
        type=int,
        default=1,
        help="minimum COV window length (default: 1)",
    )
    parser.add_argument(
        "--cov_window_max_length",
        type=int,
        default=None,
        help=(
            "optional absolute maximum COV window length; the relator margin "
            "still applies"
        ),
    )
    parser.add_argument(
        "--cov_window_relator_margin",
        type=int,
        default=1,
        help=(
            "require len(W) <= len(isolating relator) - MARGIN "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--seeded_cov_moves",
        action="store_true",
        help=(
            "queue root states with relator z w, previsit the trivial "
            "stabilization, then search with substitution and COV moves"
        ),
    )
    parser.add_argument("--ac45_moves", action="store_true")
    parser.add_argument(
        "--stable_ac_moves",
        action="store_true",
        help="alias for enabling both --change_of_variables_moves and --ac45_moves",
    )
    parser.add_argument(
        "--compare_cov",
        action="store_true",
        help="run each dataset row without COV and with COV, then compare",
    )
    parser.add_argument(
        "--compare_cov_paths",
        action="store_true",
        help=(
            "with --compare_cov, include full baseline/COV paths and actions "
            "in each JSONL row"
        ),
    )
    parser.add_argument("--show_path", action="store_true")
    parser.add_argument("--print_packed", action="store_true")
    parser.add_argument("--out_json", default=None)
    parser.add_argument(
        "--out_jsonl",
        default=None,
        help="batch-mode output path; writes one JSON object per dataset row",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    parser.add_argument("--wandb_project", default="ACSolverX-greedy")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument(
        "--wandb_mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=[],
        help="optional tags for the WandB run",
    )
    parser.add_argument(
        "--wandb_log_every",
        type=int,
        default=1,
        help="batch-mode WandB logging interval in processed rows",
    )
    parser.add_argument(
        "--wandb_window",
        type=int,
        default=100,
        help="rolling solve-rate window for WandB batch metrics",
    )
    parser.add_argument(
        "--index_summary_limit",
        type=int,
        default=10000,
        help="max number of solved/comparison indices stored in WandB summaries",
    )
    parser.add_argument(
        "--wandb_table",
        action="store_true",
        help="log a per-index results table to WandB at the end of batch runs",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def effective_move_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    change_of_variables_moves = bool(
        args.change_of_variables_moves
        or args.stable_ac_moves
        or args.seeded_cov_moves
    )
    ac45_moves = bool(args.ac45_moves or args.stable_ac_moves)
    return change_of_variables_moves, ac45_moves


def ensure_seeded_cov_capacity(args: argparse.Namespace) -> None:
    if not args.seeded_cov_moves:
        return

    if args.relators is not None:
        inferred_n = args.n_gen or max(
            len(args.relators),
            max(
                (
                    abs(code)
                    for text in args.relators
                    for code in parse_word(text)
                ),
                default=0,
            ),
        )
    else:
        probe_idx = args.start if args.start is not None else args.idx
        _, inferred_n, _ = load_dataset_presentation(
            args.dataset,
            probe_idx,
            args.n_gen,
            None,
            args.max_length,
        )
    if inferred_n != 2:
        raise SystemExit("--seeded_cov_moves requires a rank-two presentation")
    args.max_n_gen = max(args.max_n_gen or inferred_n, inferred_n + 1)


def main() -> None:
    args = parse_args()
    change_of_variables_moves, ac45_moves = effective_move_flags(args)
    if args.max_nodes <= 0:
        raise SystemExit("--max_nodes must be positive")
    if args.max_length <= 0 or args.max_total_length <= 0:
        raise SystemExit("--max_length and --max_total_length must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive")
    if args.cov_window_min_length < 1:
        raise SystemExit("--cov_window_min_length must be positive")
    if not 1 <= args.cov_window_relator_margin < args.max_length:
        raise SystemExit(
            "--cov_window_relator_margin must be between 1 and "
            "--max_length - 1"
        )
    if args.cov_window_max_length is not None:
        if not (
            args.cov_window_min_length
            <= args.cov_window_max_length
            < args.max_length
        ):
            raise SystemExit(
                "--cov_window_max_length must be at least "
                "--cov_window_min_length and less than --max_length"
            )
    custom_cov_window_bounds = (
        args.cov_window_min_length != 1
        or args.cov_window_max_length is not None
        or args.cov_window_relator_margin != 1
    )
    if custom_cov_window_bounds and not (
        change_of_variables_moves or args.compare_cov
    ):
        raise SystemExit(
            "custom COV window bounds require COV, seeded COV, "
            "or --compare_cov"
        )
    if args.wandb_log_every <= 0:
        raise SystemExit("--wandb_log_every must be positive")
    if args.wandb_window <= 0:
        raise SystemExit("--wandb_window must be positive")
    if args.index_summary_limit < 0:
        raise SystemExit("--index_summary_limit must be non-negative")
    if args.seeded_cov_moves and args.print_packed:
        raise SystemExit(
            "--print_packed cannot represent the synthetic seed action used by "
            "--seeded_cov_moves"
        )
    if args.seeded_cov_moves and args.compare_cov:
        raise SystemExit("--seeded_cov_moves cannot be combined with --compare_cov")
    if args.relators is not None and (args.all or args.start is not None or args.end is not None):
        raise SystemExit("--all/--start/--end only apply with --dataset")
    if args.compare_cov and args.relators is not None:
        raise SystemExit("--compare_cov only applies with --dataset")
    if args.compare_cov and args.dataset is None:
        raise SystemExit("--compare_cov requires --dataset")
    if args.out_jsonl is not None and args.dataset is None:
        raise SystemExit("--out_jsonl only applies with --dataset batch mode")

    ensure_seeded_cov_capacity(args)

    comparison_mode = args.compare_cov
    batch_mode = args.dataset is not None and (
        args.all or args.start is not None or args.end is not None
    )
    if args.out_jsonl is not None and not (batch_mode or comparison_mode):
        raise SystemExit("--out_jsonl only applies with --all/--start/--end or --compare_cov")
    if comparison_mode:
        run_cov_comparison(args, ac45_moves)
        return
    if batch_mode:
        run_dataset_batch(args, change_of_variables_moves, ac45_moves)
        return

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

    solver = make_solver(
        initial_flat=initial_flat,
        n_gen=n_gen,
        max_n_gen=max_n_gen,
        args=args,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
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
    wandb_run = init_wandb_run(
        args, change_of_variables_moves, ac45_moves, batch_mode=False
    )
    result = solver.solve()
    log_single_wandb(
        wandb_run,
        result,
        idx=args.idx if args.dataset is not None else None,
    )
    if wandb_run is not None:
        wandb_run.finish()
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
