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

A presentation is a flat `jnp.array` of length `2 * max_length`: two relators of
`max_length` each, zero-padded (here `n_gen = 2`, so this equals the code's
`n_gen * max_length`). The agent picks a substitution move `[i, j, k1, k2]`
packed into a single integer action index; the goal is to reach the trivial
presentation `<x, y | x, y>`.

## Layout

```
envs/                environment logic
  environment.py     gymnax base Environment (reset/step take idx, sample, probs)
  ac_s.py            ACS: the substitution environment (EnvState, step_env, reset_env)
  ac_moves.py        substitution-move implementation (setup_s_actions)
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
data/                initial-state datasets (one Python-literal presentation per line)
```

## Datasets (`data/`)

Each line is one presentation as a Python literal (a flat list of length
`2 * max_length`, i.e. two zero-padded relators). Generators are encoded as
integers: `x → 1`, `y → 2`, `x⁻¹ → -1`, `y⁻¹ → -2`, and `0` is padding.
"Length" below means the number of nonzero entries (the total word length).

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

## Training

Run from the repository root. To train and write Orbax checkpoints (needed by
the beam search) under `ppo_checkpoints/<name>/`:

```bash
python ppo_ac_s.py --ckpt_path my_run --save_every 50
```

Without `--ckpt_path`, training runs but saves nothing. Useful flags:
`--w 0` (disable wandb), `--lr`, `--ent_coef`, `--seed`, and the reward-shaping
penalties `--cycle_penalty` / `--noop_penalty` (default `0.0`). The training
dataset and `max_length` (`L = 24`) are set at the top of `make_train`.

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

The action packing/unpacking in `beam_search.py` must match `ppo_ac_s.py`
(`L = 24`).

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
