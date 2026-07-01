# AC-SolverX

**Code and datasets for *The Two-Hump Problem: Bridging the Difficulty Gap in
Mathematical Reinforcement Learning* (ICML 2026).**

- 📄 **Project page:** https://icml.cc/virtual/2026/poster/64251
- 📝 **arXiv:** https://arxiv.org/abs/2606.21611

AC-SolverX trains reinforcement-learning agents to trivialize balanced
presentations of the trivial group, the search problem at the heart of the
**Andrews–Curtis (AC) conjecture**, using composite **substitution** supermoves
together with a domain-specific **Dual-Ring Transformer** architecture. The code
is JAX-based and uses
gymnax-style environments. This repository also releases **AC-19** and **AC-1M**,
the first large-scale public datasets of AC-trivial presentations.

## Background: the Andrews–Curtis conjecture

The Andrews–Curtis conjecture concerns *balanced* presentations of the trivial
group, `<x, y | r1, r2>`, where the generators are `x` and `y` and the two
relators `r1, r2` are words that the presentation declares equal to the
identity. It states that any such presentation can be transformed into the
trivial presentation `<x, y | x, y>` by a finite sequence of three elementary
moves (the **AC-moves**):

- **(AC1)** replace a relator `r_i` by its inverse `r_i⁻¹`;
- **(AC2)** replace `r_i` by the product `r_i r_j` (`i ≠ j`);
- **(AC3)** replace `r_i` by a conjugate `w r_i w⁻¹` for any word `w`.

A presentation that can be connected to the trivial presentation this way is
called **AC-trivial**. Deciding whether a given presentation is AC-trivial is a
search problem on an enormous graph whose vertices are balanced presentations
and whose edges are single AC-moves. The conjecture has resisted proof for over
60 years; a genuine counterexample would also disprove the Generalized Property
R conjecture and yield a counterexample to the smooth 4-dimensional Poincaré
conjecture. Well-known candidate counterexamples include the Akbulut–Kirby
family `AK(n)` and the Miller–Schupp family `MS(n, w)`.

These properties make AC a clean but unforgiving testbed for reinforcement
learning: rewards are extremely sparse, and difficulty is sharply bimodal: most
presentations are either trivially easy or effectively impossible, with very few
"hard-but-solvable" instances in between. We call this the **Two-Hump
distribution**, and bridging the gap between the humps is the central challenge
this work addresses.

## What's in this work

- **Substitution supermoves.** A substitution treats both relators as cyclic
  rings and splices one into the other so that at least one letter cancels,
  collapsing many elementary AC-moves into a single action. This enables much
  larger steps and a smaller effective state space, giving our greedy search an
  ~1600× exploration-efficiency improvement over the prior baseline.
- **The Dual-Ring Transformer.** A specialized policy/value network that
  respects the cyclic symmetry of the relators (via cyclic relative positional
  encodings and cross-attention) while operating over the large substitution
  action space.
- **Targeted data generation.** Exhaustive enumeration produces **AC-19**, and
  an automorphism-based generator–solver game produces **AC-1M**, together
  populating the sparse difficulty "valley" between the two humps.
- **Concrete mathematical progress.** Our methods trivialize over 100
  previously open Miller–Schupp presentations and reduce the remaining unsolved
  cases to a smaller set of AC-equivalence classes with minimal representatives.

On the 1190-presentation Miller–Schupp benchmark, our PPO agent solves **610**
presentations versus **457** for the prior RL baseline, and consistently finds
shorter trivialization paths than classical greedy search, especially on hard
presentations.

## Quickstart

Install the dependencies for your accelerator, then evaluate the released model
on the Miller–Schupp benchmark with beam search:

```bash
# 1. Install (pick the file matching your hardware)
pip install -r requirements-cuda.txt        # NVIDIA CUDA 12
# or: pip install -r requirements-rocm.txt   # AMD ROCm

# 2. Run beam search with the pretrained 610-solve checkpoint
python beam/beam_search.py --ckpt_path 610model --beam_width 1024
```

See [Setup](#setup) for backend details and [Training](#training) to train your
own agent from scratch.

## Code overview

A presentation is a flat `jnp.array` of length `max_n_gen * max_length`:
`max_n_gen` relator slots, each zero-padded to `max_length`. The environment
tracks the active generator count separately as `state.n_gen`, so stable moves
can add or remove generators while the observation shape stays fixed. By
default, `max_n_gen = n_gen = 2`, preserving the released checkpoint setup.

By default the environment exposes substitution supermoves. Additional action
families are configured by explicit flags:

- `change_of_variables_moves=True` enables the finite change-of-variables
  supermove.
- `ac45_moves=True` enables AC4 add-generator and AC5 delete-generator.

For a change-of-variables run, use
`change_of_variables_moves=True, ac45_moves=False`.

## Layout

```
envs/                environment logic
  environment.py     gymnax base Environment (reset/step take idx, sample, probs)
  ac_s.py            ACS: the substitution environment (EnvState, step_env, reset_env)
  ac_moves.py        substitution, AC4/AC5, and change-of-variables moves
  utils.py           presentation padding/length helpers
  int_box.py         minimal integer Box space
network.py           RelativeDualRingActorCritic (used by PPO) and DualRingActorCritic
wrappers.py          LogWrapper, NormalizeVecReward, LogPathsProbsS
ppo_ac_s.py          PPO training entry point
greedy_search.ipynb  greedy search with substitutions (GS-Sub); standalone (numpy + numba)
beam/
  beam_search.py     beam search that loads a trained checkpoint
scripts/
  check_checkpoint_paths.py  validate stored paths in a checkpoint's solve_data
  stable_cov_demo.py         worked AK(3) stable change-of-variables move
data/                initial-state datasets (one Python-literal presentation per line)
```

## Datasets (`data/`)

Each line is one presentation as a Python literal. The released datasets are
flat lists of length `2 * max_length`, i.e. two zero-padded relators. Generators
are encoded as integers: `x → 1`, `y → 2`, `x⁻¹ → -1`, `y⁻¹ → -2`, and `0` is
padding. "Length" below means the number of nonzero entries (the total word
length). For custom `n`-generator data, store `n` zero-padded relator slots per
line and instantiate `ACS(n_gen=n, max_n_gen=...)`.

For example, the first line of `1190MS.txt` begins
`[-2, -2, -1, 2, 1, 0, …]`: the first relator is `y⁻¹ y⁻¹ x⁻¹ y x`, with the rest
of its 24-slot block zero-padded, followed by the second relator's block.

- **`1190MS.txt`**: the 1190-presentation Miller–Schupp benchmark (the
  validation set from prior work), stored in **canonical form**: each relator is
  the lexicographically minimal element among the cyclic rotations of itself and
  of its inverse, and the two relators are ordered lexicographically. This gives
  a unique representative per equivalence class under cyclic rotation, inversion,
  and relator ordering.
- **`AC19_extended.txt`**: the default training set (156,762 presentations).
  Its structure:
    - the **first 634** lines are the first 634 presentations of `1190MS.txt`;
    - the remaining lines are **AC19** (presentations of length ≤ 19) together
      with some other longer presentations (length > 19).
  (Of the 156,762: 140,874 have length ≤ 19 and 15,888 have length > 19.)
- **`AC19.txt`**: `AC19_extended.txt` with the first 634 presentations removed
  and then filtered to length ≤ 19 (140,535 presentations).
- **`AC1M.txt.gz`**: a larger set of 1,136,154 presentations, shipped
  gzip-compressed (246 MB uncompressed). Decompress before using it as a
  dataset:
  ```bash
  gunzip -k data/AC1M.txt.gz   # -> data/AC1M.txt, then use stem "AC1M"
  ```

`ACS(initial_states_file=...)` takes the bare stem (no `data/` prefix, no
`.txt`).

## Setup

Dependencies are split so the JAX stack can target either backend. The
platform-independent packages live in `requirements.txt`; install it together
with one platform file:

```bash
pip install -r requirements-rocm.txt   # AMD ROCm
pip install -r requirements-cuda.txt   # NVIDIA CUDA 12
```

Each platform file does `-r requirements.txt` and then pins JAX 0.6.0 plus the
matching `jaxlib`/plugin. For the ROCm build, make sure a compatible ROCm
runtime is available in your environment before running.

## Greedy search

`greedy_search.ipynb` implements **GS-Sub**, the classical greedy (best-first)
search over substitution moves used as a baseline throughout the paper. It is
self-contained and depends only on `numpy` and `numba`: no JAX, no GPU, and no
checkpoint required. The notebook:

- maintains presentations in **canonical form** (minimal cyclic rotation via
  Booth's algorithm, inversion, and relator ordering);
- expands states in order of total length (`|r1| + |r2|`), exploring shorter
  presentations first;
- recognizes the trivial presentation and known potential counterexamples
  (`AK(n)`, the length-14 pairs).

Open it and run the cells. The final cell is a worked example on a
Miller–Schupp presentation:

```python
r1, r2 = MS(3, 'YXyxy')
solver = ACRelatorSolver(r1, r2, max_nodes=10000, max_len=20)
path, nodes, seen = solver.solve()
```

Key knobs are `max_nodes` (search budget) and `max_len` (the largest relator
length the search will consider, smaller is faster). Relators are written as
strings over `x, X, y, Y` (where `X = x⁻¹`, `Y = y⁻¹`).

## Optional Stable Moves

The move families are:

- **Change of variables**: choose a cyclic subword `W` in one relator,
  introduce `z = W` (or `z = W^-1`), use the complementary part of that relator
  to isolate one old generator, then remove that old generator and substitute
  the solution into the remaining active relators and the defining relator for
  `z`.
- **AC4 add generator**: append a new generator `x_{n+1}` and a trivial relator
  `x_{n+1}` when `state.n_gen < max_n_gen`.
- **AC5 delete generator**: remove generator `x_i` when exactly one active
  relator contains `x_i`, and that relator is the singleton `x_i` or `x_i^-1`.
  Higher generator labels and relator slots are compacted after deletion.

These actions are opt-in. For a change-of-variables run, set
`change_of_variables_moves=True` and `ac45_moves=False`. With `M = max_n_gen`
and `L = max_length`, generic substitution mode uses
`M * (M - 1) * 2 * L * L` actions. The historical `M = 2` layout is kept at
`4 * L * L` actions. `change_of_variables_moves=True` appends
`M * M * 2 * L * L` actions. `ac45_moves=True` appends one AC4 add action and
`M` AC5 delete actions.

Programmatic usage for a generic fixed capacity:

```python
from envs.ac_s import ACS

env = ACS(n_gen=3, max_n_gen=6, max_length=32,
          max_steps_in_episode=200,
          initial_states_file="my_3gen_dataset",
          change_of_variables_moves=True,
          ac45_moves=False)
```

Environment action vectors use branch IDs, not packed policy indices. AC4/AC5
action vectors are relevant when `ac45_moves=True`:

```python
from envs.utils import add_generator_branch, delete_generator_branch

M = 6

# AC4 add generator
[add_generator_branch(M, change_of_variables_moves=True), 0, 0, 0]

# AC5 delete generator with zero-based generator index delete_gen
[delete_generator_branch(M, change_of_variables_moves=True), delete_gen, 0, 0]
```

If AC4/AC5 is enabled without change-of-variables, call the branch helpers
with `change_of_variables_moves=False`.

For change-of-variables, the branch starts after the substitution branches:

```python
from envs.utils import change_of_variables_branch

[change_of_variables_branch(remove_gen, iso_relator, M),
 z_inverse, z_start, z_len - 1]
```

- `remove_gen`: zero-based generator slot to replace by the new variable `z`.
- `iso_relator`: active relator used to isolate the old generator.
- `z_inverse`: `0` means `z = W`; `1` means `z = W^-1`.
- `z_start`: cyclic start index of `W` inside `iso_relator`.
- `z_len - 1`: encoded subword length, so `z_len` ranges from `1` to `L`.

Invalid stable moves leave the state unchanged. This includes trying to add
past `max_n_gen`, trying to delete a generator that is not isolated by a
trivial relator, choosing an inactive relator/generator, or overflowing
`max_length` after free and cyclic reduction.

Run the worked `AK(3)` example from the change-of-variables note:

```bash
python scripts/stable_cov_demo.py
```

Change-of-variables programmatic setup:

```python
from envs.ac_s import ACS
from network import RelativeDualRingActorCritic

env = ACS(n_gen=2, max_n_gen=2, max_length=24, max_steps_in_episode=150,
          initial_states_file="AC19_extended",
          change_of_variables_moves=True,
          ac45_moves=False)
network = RelativeDualRingActorCritic(activation="gelu",
                                      change_of_variables_moves=True)
```

Pack and decode change-of-variables paths with the same flag:

```python
from envs.utils import encode_action, decode_path

packed = encode_action([2, 0, 0, 2], max_length=24,
                       change_of_variables_moves=True)
decoded = decode_path([packed], max_length=24,
                      change_of_variables_moves=True)
```

`RelativeDualRingActorCritic`, `ppo_ac_s.py`, and `beam/beam_search.py` remain
two-relator model code. They explicitly require `n_gen = max_n_gen = 2`. The
environment and move layer support general `n` and `max_n_gen`; training a
neural policy for `n > 2` needs an `n`-relator architecture.

Do not pass `--change_of_variables_moves` or `--ac45_moves` when evaluating the released `610model`: that
checkpoint was trained with the substitution-only policy head. Train a matching
stable policy head before running stable beam search.

## Training

Run from the repository root. To train and write Orbax checkpoints (needed by
the beam search) under `ppo_checkpoints/<name>/`:

```bash
python ppo_ac_s.py --ckpt_path my_run --save_every 50
```

Without `--ckpt_path`, training runs but saves nothing. To train with
change-of-variables moves in the current two-relator policy, add
`--change_of_variables_moves`:

```bash
python ppo_ac_s.py --change_of_variables_moves --ckpt_path cov_run --save_every 50
```

AC4/AC5 remains disabled unless you add `--ac45_moves`.

Useful flags: `--w 0` (disable wandb), `--lr`, `--ent_coef`, `--seed`, and the
reward-shaping penalties `--cycle_penalty` / `--noop_penalty` (default `0.0`).
The training dataset and `max_length` (`L = 24`) are set at the top of
`make_train`.

The network is `RelativeDualRingActorCritic` (relative-position attention,
cyclic relators). `DualRingActorCritic` (absolute positional encoding) is
provided in `network.py` as an alternative.

## Beam search

Loads a trained checkpoint and runs beam search per presentation. Run from the
repository root so `data/` and `ppo_checkpoints/` resolve:

```bash
python beam/beam_search.py --ckpt_path my_run --beam_width 1024 \
    --start 0 --end 634 --out_csv beam_paths.csv
```

For checkpoints trained with change-of-variables moves, pass the same flag:

```bash
python beam/beam_search.py --change_of_variables_moves --ckpt_path cov_run \
    --beam_width 1024 --start 0 --end 634 --out_csv cov_beam_paths.csv
```

Beam search checks the saved training config and exits if the
move-family flags do not match the checkpoint. The action packing/unpacking in
`beam_search.py` must match `ppo_ac_s.py` (`L = 24`).

## Pretrained checkpoint

A trained model is provided under `ppo_checkpoints/610model/` (latest step
1000) that solves 610 of the first 634 presentations (the `1190MS.txt`-derived
benchmark set at the start of `AC19_extended.txt`). Use it directly with the
beam search or the path validator:

```bash
python beam/beam_search.py --ckpt_path 610model --beam_width 1024
python scripts/check_checkpoint_paths.py --ckpt_path 610model --max_paths 10
```

## License

- **Code** is released under the [MIT License](LICENSE).
- **Datasets** (`data/`, including AC-19 and AC-1M) are released under
  [CC BY 4.0](LICENSE-DATA).

Both licenses permit free use, modification, and redistribution, including
commercially, as long as you give appropriate credit. We ask that you
attribute this work by citing the paper below.

## Citation

If you use this code or the AC-19 / AC-1M datasets, please cite:

```bibtex
@inproceedings{fagan2026twohump,
  title     = {The Two-Hump Problem: Bridging the Difficulty Gap in Mathematical Reinforcement Learning},
  author    = {Fagan, Lucas and Tarquini, Michele and Shehper, Ali and Manko, Maksymilian and Gruen, Angus and Huang, Coco and Butbaia, Giorgi and Passaro, Davide and Gukov, Sergei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
