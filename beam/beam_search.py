"""Beam-search evaluation against an Orbax checkpoint written by ppo_ac_s.py.

For each presentation in `[--start, --end)` of the chosen dataset, runs a beam
search of width B using the loaded actor/critic. Scoring is
`cum_log_prob + log_softmax + alpha*value` with optional Gumbel noise scaled by
a linearly decaying temperature schedule. Stops the search for a presentation
as soon as any beam reaches a trivial presentation.

Run from the repository root, e.g.:
    python beam/beam_search.py --ckpt_path my_run --beam_width 1024
"""

import os
import sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

# Allow `python beam/beam_search.py` from the repo root: put the repo root
# (parent of beam/) on sys.path so `envs` and `network` import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from envs.ac_s import ACS
from network import RelativeDualRingActorCritic
from envs.utils import decode_action_jax, decode_path as decode_packed_path

jax.config.update("jax_default_matmul_precision", "float32")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="folder under ppo_checkpoints/ to restore from")
    p.add_argument("--params_only_checkpoint", action="store_true",
                   help="load checkpoints saved with ppo_ac_s.py "
                        "--params_only_checkpoint")
    p.add_argument("--ckpt_step", type=int, default=-1,
                   help="-1 = latest step in the manager")
    p.add_argument("--dataset", type=str, default="AC19_extended",
                   help="dataset stem (data/<stem>.txt) the beam env runs on")
    p.add_argument("--training_dataset", type=str, default="AC19_extended",
                   help="stem of the dataset the model was TRAINED on; "
                        "needed to size the saved solve_data when --dataset "
                        "differs (e.g. when running on a filtered subset)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=634)
    p.add_argument("--beam_width", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="value-head coefficient in scoring")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Gumbel noise scale at t=0")
    p.add_argument("--temp_end", type=float, default=0.0,
                   help="Gumbel noise scale at t=max_steps-1 (linear schedule)")
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--change_of_variables_moves", action="store_true",
                   help="enable the change-of-variables action head; "
                        "must match the checkpoint training config")
    p.add_argument("--ac45_moves", action="store_true",
                   help="enable AC4/AC5 action head; must match the checkpoint "
                        "training config")
    p.add_argument("--stable_ac_moves", action="store_true",
                   help="deprecated alias for enabling both "
                        "--change_of_variables_moves and --ac45_moves")
    p.add_argument("--max_n_gen", type=int, default=2,
                   help="maximum generator capacity; beam search with the "
                        "two-ring checkpoint architecture currently supports only 2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=str, default="beam_paths.csv")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")
    if args.max_n_gen != 2:
        raise SystemExit(
            "beam/beam_search.py uses the two-ring transformer and currently "
            "requires --max_n_gen 2. Use envs.ac_s.ACS directly for generic "
            "n-generator stable AC moves."
        )
    change_of_variables_moves = args.change_of_variables_moves or args.stable_ac_moves
    ac45_moves = args.ac45_moves or args.stable_ac_moves

    L = 24  # max_length; the ACS action-packing constant
    env = ACS(n_gen=2, max_n_gen=args.max_n_gen,
              max_length=L, max_steps_in_episode=args.max_steps,
              is_reward_sparse=False,
              initial_states_file=args.dataset,
              change_of_variables_moves=change_of_variables_moves,
              ac45_moves=ac45_moves)
    env_params = env.default_params
    network = RelativeDualRingActorCritic(
        activation=args.activation,
        change_of_variables_moves=change_of_variables_moves,
        ac45_moves=ac45_moves,
    )
    n_actions = env.num_actions
    B = args.beam_width
    T = args.max_steps

    # Dummy params for Orbax. The solve_data shapes depend on the *training*
    # dataset size and the training NUM_STEPS, which we recover from the saved
    # config below.
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    obs_shape = env.observation_space(env_params).shape
    dummy_params = network.init(init_rng, jnp.zeros((1, *obs_shape)))

    # Training dataset size = line count of its source file
    train_src_path = os.path.join("data", f"{args.training_dataset}.txt")
    with open(train_src_path, "r") as f:
        train_num_states = sum(1 for _ in f)

    ckpt_path_abs = os.path.join(os.getcwd(), "ppo_checkpoints", args.ckpt_path)
    ckpt_item_names = (
        ("params", "config")
        if args.params_only_checkpoint
        else ("params", "solve_data", "config")
    )
    mngr = ocp.CheckpointManager(
        ckpt_path_abs,
        item_names=ckpt_item_names,
    )
    step_to_load = mngr.latest_step() if args.ckpt_step < 0 else args.ckpt_step
    if step_to_load is None:
        raise SystemExit(f"No checkpoints found at {ckpt_path_abs}")
    print(f"Restoring checkpoint step {step_to_load} from {ckpt_path_abs}")

    # Stage 1: load config to get training NUM_STEPS (sizes best_paths).
    cfg_restored = mngr.restore(
        step_to_load,
        args=ocp.args.Composite(config=ocp.args.JsonRestore({})),
    )
    ckpt_cov = bool(cfg_restored.config.get(
        "CHANGE_OF_VARIABLES_MOVES",
        cfg_restored.config.get("STABLE_AC_MOVES", False),
    ))
    ckpt_ac45 = bool(cfg_restored.config.get(
        "AC45_MOVES",
        cfg_restored.config.get("STABLE_AC_MOVES", False),
    ))
    if ckpt_cov != change_of_variables_moves or ckpt_ac45 != ac45_moves:
        raise SystemExit(
            "Checkpoint move flags do not match beam flags: "
            f"checkpoint CHANGE_OF_VARIABLES_MOVES={ckpt_cov}, AC45_MOVES={ckpt_ac45}; "
            f"requested CHANGE_OF_VARIABLES_MOVES={change_of_variables_moves}, "
            f"AC45_MOVES={ac45_moves}."
        )
    train_num_steps = int(cfg_restored.config["NUM_STEPS"])

    dummy_solve_data = {
        "solved_idx": jnp.zeros(train_num_states, dtype=jnp.bool_),
        "path_lengths": jnp.zeros(train_num_states, dtype=jnp.int32),
        "best_paths": jnp.zeros((train_num_states, train_num_steps), dtype=jnp.int32),
    }

    # Stage 2: load params, and solve_data when the checkpoint contains it.
    if args.params_only_checkpoint:
        restored = mngr.restore(
            step_to_load,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(dummy_params),
            ),
        )
    else:
        restored = mngr.restore(
            step_to_load,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(dummy_params),
                solve_data=ocp.args.StandardRestore(dummy_solve_data),
            ),
        )
    params = restored.params
    if args.params_only_checkpoint:
        train_solved = np.zeros((train_num_states,), dtype=bool)
        train_path_lengths = np.full((train_num_states,), -1, dtype=np.int32)
    else:
        train_solved = np.asarray(restored.solve_data["solved_idx"])
        train_path_lengths = np.asarray(restored.solve_data["path_lengths"])

    # If --dataset is a filtered subset produced by make_unsolved_dataset.py,
    # a sidecar `_orig_indices.txt` maps new beam indices -> original training
    # indices. Pick it up if present so the output references original indices.
    orig_idx_path = os.path.join("data", f"{args.dataset}_orig_indices.txt")
    if os.path.exists(orig_idx_path):
        with open(orig_idx_path, "r") as f:
            orig_indices = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded original-index map from {orig_idx_path} "
              f"({len(orig_indices)} entries)")
    else:
        orig_indices = None

    # Linear temperature schedule over T steps
    temp_schedule = jnp.linspace(args.temperature, args.temp_end, T)

    alpha = args.alpha

    # Fixed random vector used to hash state.x into a single int for dedup.
    # Same state -> same hash; different states almost never collide
    # (collision prob ~ B^2 / 2^32 ~ 4e-3 at B=4096; collisions only cost a
    # slot, never correctness on solve detection).
    obs_len = obs_shape[0]
    hash_vec = jax.random.randint(
        jax.random.PRNGKey(0xACABDED),
        (obs_len,),
        minval=-(2**31), maxval=2**31 - 1, dtype=jnp.int32,
    )

    HASH_SENTINEL = jnp.iinfo(jnp.int32).max
    GLOBAL_VISIT_CAP = B * T   # max hashes we will ever retain across the run

    @jax.jit
    def beam_step(states, alive, cum_log_prob, action_seqs, visited_sorted, rng, t):
        rng, noise_rng, step_rng = jax.random.split(rng, 3)

        # Policy + value over current beam
        pi, value = network.apply(params, states.x)            # logits: (B, A), value: (B,)
        log_probs = jax.nn.log_softmax(pi.logits, axis=-1)     # (B, A)
        action_scores = log_probs + alpha * value[:, None]     # (B, A)

        # Gumbel noise (no-op when temp_schedule[t] == 0)
        u = jax.random.uniform(noise_rng, action_scores.shape, minval=1e-6, maxval=1.0 - 1e-6)
        gumbel = -jnp.log(-jnp.log(u))
        action_scores = action_scores + temp_schedule[t] * gumbel

        # Dead beams contribute -inf so top_k ignores them.
        action_scores = jnp.where(alive[:, None], action_scores, -jnp.inf)

        candidate_scores = cum_log_prob[:, None] + action_scores  # (B, A)
        flat_scores = candidate_scores.reshape(-1)                # (B*A,)
        top_vals, top_idx = jax.lax.top_k(flat_scores, B)

        parent = top_idx // n_actions
        flat_action = top_idx % n_actions

        action_vec = decode_action_jax(
            flat_action, L, change_of_variables_moves, ac45_moves,
            max_n_gen=args.max_n_gen
        )

        parent_states = jax.tree.map(lambda v: v[parent], states)
        step_rngs = jax.random.split(step_rng, B)
        _, new_states, _, dones, info = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(step_rngs, parent_states, action_vec, env_params)

        new_action_seqs = action_seqs[parent].at[:, t].set(flat_action)

        # Noop detection: env silently no-ops moves that would push a relator
        # past max_length. Without this kill, beam search rewards such noops
        # because they keep racking up high log_prob without changing state,
        # locking the population onto a single stuck state.
        noop_mask = jnp.all(parent_states.x == new_states.x, axis=-1)
        parent_alive = alive[parent]
        terminated = info["terminated"] & parent_alive
        new_alive = parent_alive & (~dones) & (top_vals > -1e30) & (~noop_mask)
        new_cum_log_prob = jnp.where(new_alive, top_vals, -jnp.inf)

        # State deduplication: hash each beam's new state, keep only the
        # highest-cumlp beam per unique hash. Without this, multiple parents
        # routinely converge on the same successor state and top-K wastes
        # slots on duplicates.
        hashes = jnp.sum(new_states.x.astype(jnp.int32) * hash_vec[None, :], axis=1)
        # Sort by (hash ascending, -cumlp ascending) -> within hash group the
        # highest cumlp comes first. Dead beams (cumlp=-inf) come last.
        sort_order = jnp.lexsort(jnp.stack([-new_cum_log_prob, hashes]))
        sorted_hashes = hashes[sort_order]
        is_first = jnp.concatenate(
            [jnp.array([True]), sorted_hashes[1:] != sorted_hashes[:-1]]
        )
        keep = jnp.zeros(B, dtype=jnp.bool_).at[sort_order].set(is_first)
        new_alive = new_alive & keep
        new_cum_log_prob = jnp.where(new_alive, new_cum_log_prob, -jnp.inf)

        # Global visited check: kill beams whose new state was already visited
        # at any earlier step (by any beam). Implemented as binary search into
        # a sorted-with-sentinel buffer.
        idx_in_vis = jnp.searchsorted(visited_sorted, hashes)
        # Clamp index for safe gather (sentinel rows return GLOBAL_VISIT_CAP)
        idx_clamped = jnp.minimum(idx_in_vis, GLOBAL_VISIT_CAP - 1)
        already_visited = (visited_sorted[idx_clamped] == hashes)
        new_alive = new_alive & (~already_visited)
        new_cum_log_prob = jnp.where(new_alive, new_cum_log_prob, -jnp.inf)

        # Update visited_sorted: append surviving alive hashes (others become
        # sentinel), then re-sort. Truncate to keep the buffer size constant.
        new_entries = jnp.where(new_alive, hashes, HASH_SENTINEL)
        merged = jnp.concatenate([visited_sorted, new_entries])
        new_visited_sorted = jnp.sort(merged)[:GLOBAL_VISIT_CAP]

        return (new_states, new_alive, new_cum_log_prob, new_action_seqs,
                terminated, new_visited_sorted, rng)

    def decode_path(flat_actions):
        return decode_packed_path(
            flat_actions, L,
            change_of_variables_moves=change_of_variables_moves,
            ac45_moves=ac45_moves,
            max_n_gen=args.max_n_gen,
        )

    results = []
    t0 = time.time()
    n_total = args.end - args.start
    for p_idx in range(args.start, args.end):
        rng, reset_rng = jax.random.split(rng)
        # Single env reset: idx=p_idx, sample=False -> EnvState whose x is init_states[p_idx]
        _, init_state = env.reset_env(
            reset_rng, env_params,
            idx=jnp.int32(p_idx),
            sample=jnp.bool_(False),
            probs=None,
        )

        # Broadcast init state across the beam. EnvState may contain plain
        # Python scalars (e.g. time=0), so coerce to jnp arrays first.
        def _bcast(v):
            arr = jnp.asarray(v)
            return jnp.broadcast_to(arr[None], (B,) + arr.shape)
        states = jax.tree.map(_bcast, init_state)
        alive = jnp.ones(B, dtype=jnp.bool_)
        # Only beam 0 is "real" at t=0; the rest are -inf so they don't contribute
        # to the first expansion's top_k.
        cum_log_prob = jnp.full((B,), -jnp.inf, dtype=jnp.float32).at[0].set(0.0)
        action_seqs = jnp.full((B, T), -1, dtype=jnp.int32)

        # Global visited buffer: pre-allocated, padded with HASH_SENTINEL.
        # Seed with the init state's hash so beams can't loop back to it.
        init_hash = jnp.sum(
            jnp.asarray(init_state.x, dtype=jnp.int32) * hash_vec
        )
        visited_sorted = jnp.full((GLOBAL_VISIT_CAP,), HASH_SENTINEL, dtype=jnp.int32)
        visited_sorted = visited_sorted.at[0].set(init_hash)
        visited_sorted = jnp.sort(visited_sorted)

        solved = False
        solved_len = -1
        solved_path = []
        for t_step in range(T):
            rng, step_rng = jax.random.split(rng)
            states, alive, cum_log_prob, action_seqs, terminated, visited_sorted, _ = beam_step(
                states, alive, cum_log_prob, action_seqs, visited_sorted, step_rng, jnp.int32(t_step)
            )
            if bool(terminated.any()):
                term_np = np.asarray(terminated)
                beam_idx = int(np.argmax(term_np))  # first True
                solved = True
                solved_len = t_step + 1
                solved_path = np.asarray(action_seqs[beam_idx, :solved_len]).tolist()
                break

        # Resolve back to the original training-dataset index if a sidecar
        # map is available; otherwise the beam index *is* the training index.
        orig_p = orig_indices[p_idx] if orig_indices is not None else p_idx
        train_was_solved = bool(train_solved[orig_p]) if orig_p < len(train_solved) else False
        train_len = int(train_path_lengths[orig_p]) if orig_p < len(train_path_lengths) else -1
        label = f"[{p_idx} -> orig {orig_p}]" if orig_indices is not None else f"[{p_idx}]"
        if solved:
            print(f"{label} SOLVED in {solved_len}  (training-solved={train_was_solved}, train_len={train_len})", flush=True)
        else:
            print(f"{label} not solved              (training-solved={train_was_solved}, train_len={train_len})", flush=True)

        results.append({
            "presentation_idx": p_idx,
            "orig_presentation_idx": orig_p,
            "solved": solved,
            "path_length": solved_len,
            "path": solved_path,
            "path_decoded": decode_path(solved_path) if solved else [],
            "train_solved": train_was_solved,
            "train_path_length": train_len if train_was_solved else -1,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    n_solved = int(df["solved"].sum())
    elapsed = time.time() - t0
    print(f"\nDone. {n_solved}/{n_total} solved (beam_width={B}, T={T}, "
          f"alpha={alpha}, temp={args.temperature}->{args.temp_end}) "
          f"in {elapsed:.1f}s")
    print(f"Wrote {args.out_csv}")

    # Delta vs training: presentations we solve that training did not.
    new_solves = df[(df["solved"]) & (~df["train_solved"])]
    if len(new_solves) > 0:
        print(f"Beam found {len(new_solves)} solves not in training checkpoint: "
              f"{new_solves['presentation_idx'].tolist()}")


if __name__ == "__main__":
    main()
