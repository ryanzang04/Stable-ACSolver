"""Validate stored PPO paths from an Orbax checkpoint's solve_data.

Restores solved_idx / path_lengths / best_paths from a checkpoint written by
ppo_ac_s.py (with --ckpt_path) and replays each solving path in the ACS env to
confirm it reaches the trivial presentation. By default only the first
--max_paths solved paths are checked.

Run from the repository root, e.g.:
    python scripts/check_checkpoint_paths.py --ckpt_path ppo_test --max_paths 10
"""

import os
import sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

# Allow `python scripts/check_checkpoint_paths.py` from the repo root: put the
# repo root (parent of scripts/) on sys.path so `envs` imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import jax.numpy as jnp
import orbax.checkpoint as ocp
from envs.utils import check_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="folder under ppo_checkpoints/ to restore from")
    p.add_argument("--ckpt_step", type=int, default=-1,
                   help="-1 = latest step in the manager")
    p.add_argument("--dataset", type=str, default="AC19_extended",
                   help="training dataset stem (data/<stem>.txt) the stored "
                        "paths index into; must match what the run trained on")
    p.add_argument("--max_length", type=int, default=24)
    p.add_argument("--max_paths", type=int, default=10,
                   help="number of solved paths to validate (default 10; "
                        "-1 = all)")
    return p.parse_args()


def main():
    args = parse_args()

    # Number of initial states = line count of the training dataset file; this
    # sizes the saved solve_data arrays.
    src = os.path.join("data", f"{args.dataset}.txt")
    with open(src, "r") as f:
        num_states = sum(1 for _ in f)

    ckpt_abs = os.path.join(os.getcwd(), "ppo_checkpoints", args.ckpt_path)
    mngr = ocp.CheckpointManager(
        ckpt_abs, item_names=("params", "solve_data", "config"),
    )
    step = mngr.latest_step() if args.ckpt_step < 0 else args.ckpt_step
    if step is None:
        raise SystemExit(f"No checkpoints found at {ckpt_abs}")
    print(f"Restoring solve_data from {ckpt_abs} step {step}")

    # NUM_STEPS sizes the best_paths width.
    cfg = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore({})))
    num_steps = int(cfg.config["NUM_STEPS"])
    change_of_variables_moves = bool(cfg.config.get(
        "CHANGE_OF_VARIABLES_MOVES",
        cfg.config.get("STABLE_AC_MOVES", False),
    ))
    ac45_moves = bool(cfg.config.get(
        "AC45_MOVES",
        cfg.config.get("STABLE_AC_MOVES", False),
    ))
    n_gen = int(cfg.config.get("N_GEN", 2))
    max_n_gen = int(cfg.config.get("MAX_N_GEN", n_gen))

    dummy_solve_data = {
        "solved_idx": jnp.zeros(num_states, dtype=jnp.bool_),
        "path_lengths": jnp.zeros(num_states, dtype=jnp.int32),
        "best_paths": jnp.zeros((num_states, num_steps), dtype=jnp.int32),
    }
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(solve_data=ocp.args.StandardRestore(dummy_solve_data)),
    )
    sd = restored.solve_data
    n_solved = int(jnp.count_nonzero(sd["solved_idx"]))
    print(f"checkpoint reports {n_solved} solved presentations "
          f"(change_of_variables_moves={change_of_variables_moves}, "
          f"ac45_moves={ac45_moves}, n_gen={n_gen}, "
          f"max_n_gen={max_n_gen})")

    max_paths = None if args.max_paths < 0 else args.max_paths
    failures = check_paths(
        sd["solved_idx"], sd["path_lengths"], sd["best_paths"],
        initial_states_file=args.dataset,
        max_length=args.max_length,
        max_paths=max_paths,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
        n_gen=n_gen,
        max_n_gen=max_n_gen,
    )
    if failures:
        raise SystemExit(f"{len(failures)} path(s) FAILED validation: {failures}")
    print("OK: all checked paths valid")


if __name__ == "__main__":
    main()
