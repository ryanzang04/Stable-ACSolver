"""PPO training for the substitution (ACS) Andrews-Curtis environment.

Trains a RelativeDualRingActorCritic with PPO. The outer training loop is a
plain Python `for` over a jitted single-update step so that Orbax can write
checkpoints between updates (Orbax can't run inside a jax.lax.scan). Pass
`--ckpt_path NAME` to save checkpoints under `ppo_checkpoints/NAME/`; the beam
search (beam/beam_search.py) restores from those.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import time
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from typing import NamedTuple
from flax.training.train_state import TrainState
from envs.ac_s import ACS
from wrappers import LogWrapper, NormalizeVecReward, LogPathsProbsS
from network import RelativeDualRingActorCritic
from envs.utils import decode_action_jax, encode_action_jax

jax.config.update("jax_default_matmul_precision", "float32")


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    if config.get("N_GEN", 2) != 2 or config.get("MAX_N_GEN", 2) != 2:
        raise ValueError(
            "ppo_ac_s.py uses the two-ring transformer and currently requires "
            "N_GEN=2 and MAX_N_GEN=2. Use envs.ac_s.ACS directly for generic "
            "n-generator stable AC moves."
        )
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    L = 24
    env = ACS(n_gen=config.get("N_GEN", 2), max_n_gen=config.get("MAX_N_GEN", 2),
              max_length=L, max_steps_in_episode=int(config["NUM_STEPS"]),
              is_reward_sparse=False,
              initial_states_file='AC19_extended',
              cycle_penalty=config.get("CYCLE_PENALTY", 0.0),
              noop_penalty=config.get("NOOP_PENALTY", 0.0),
              change_of_variables_moves=config.get("CHANGE_OF_VARIABLES_MOVES", False),
              ac45_moves=config.get("AC45_MOVES", False))
    env_params = env.default_params
    env = LogWrapper(env)
    env = NormalizeVecReward(env, config["GAMMA"])
    env = LogPathsProbsS(env, config["NUM_ENVS"])

    network = RelativeDualRingActorCritic(
        activation=config["ACTIVATION"],
        change_of_variables_moves=config.get("CHANGE_OF_VARIABLES_MOVES", False),
        ac45_moves=config.get("AC45_MOVES", False),
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        obs_shape = env.observation_space(env_params).shape
        init_x = jnp.zeros((1, *obs_shape))
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        parallel_idx = jnp.tile(jnp.arange(len(env.init_states)),
                                (config["NUM_ENVS"] // len(env.init_states),))
        parallel_idx = jnp.concatenate([parallel_idx,
                                        jnp.arange(config["NUM_ENVS"] % len(env.init_states))])
        parallel_sample = jnp.full((config["NUM_ENVS"],), True)
        parallel_sample = parallel_sample.at[:634].set(False)
        obsv, env_state = env.reset(reset_rng, env_params, parallel_idx, parallel_sample)

        # TRAIN LOOP — single update step. The outer Python loop drives this so
        # Orbax can save checkpoints between updates (Orbax can't run inside a
        # jax.lax.scan).
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                sample = pi.sample(seed=_rng)
                action = decode_action_jax(
                    sample, L,
                    config.get("CHANGE_OF_VARIABLES_MOVES", False),
                    config.get("AC45_MOVES", False),
                    max_n_gen=config.get("MAX_N_GEN", 2),
                )
                log_prob = pi.log_prob(sample)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done, transition.value, transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        sample = encode_action_jax(
                            traj_batch.action,
                            L,
                            config.get("CHANGE_OF_VARIABLES_MOVES", False),
                            config.get("AC45_MOVES", False),
                            max_n_gen=config.get("MAX_N_GEN", 2),
                        )
                        log_prob = pi.log_prob(sample)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            adv = update_state[2]

            values_flat = traj_batch.value.flatten()
            targets_flat = targets.flatten()
            explained_var = 1 - jnp.var(values_flat - targets_flat) / jnp.var(targets_flat)

            if config.get("DEBUG"):
                def print_callback(info, env_state):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}, num solved={int(jnp.count_nonzero(env_state.solved_idx))}")
                jax.debug.callback(print_callback, metric, env_state)

            if config.get("WANDB_MODE", "disabled") == "online":
                def wandb_callback(info, loss_info, adv, traj_batch, env_state, explained_var):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    solved_ids = (env_state.solved_idx).nonzero()[0]
                    num_solved = len(solved_ids)

                    num_solved_interesting = sum(solved_ids <= 634)
                    largest_solved_idx = -1 if num_solved == 0 else int(solved_ids.max())

                    cycle_rate = float(traj_batch.info["cycle_hit"].mean()) if "cycle_hit" in traj_batch.info else 0.0
                    noop_rate = float(traj_batch.info["noop_hit"].mean()) if "noop_hit" in traj_batch.info else 0.0
                    wandb.log({"global_step": timesteps[-1],
                               "entropy_loss": float(loss_info[1][2].mean()),
                               "value_loss": float(loss_info[1][0].mean()),
                               "policy_loss": float(loss_info[1][1].mean()),
                               "adv_mean": float(adv.mean()),
                               "adv_std": float(adv.std()),
                               "mean_length": float(traj_batch.info["length"].mean()),
                               "min_length": float(traj_batch.info["length"].min()),
                               "max_length": float(traj_batch.info["length"].max()),
                               "num_solved": num_solved,
                               "num_solved_interesting": num_solved_interesting,
                               "largest_solved_idx": largest_solved_idx,
                               "recently_solved_no_duplicates": (info["terminated"].sum(axis=0) != 0).sum(),
                               "recently_solved_with_duplicates": info["terminated"].sum(axis=0).sum(),
                               "explained_var": float(explained_var),
                               "cycle_rate": cycle_rate,
                               "noop_rate": noop_rate})

                jax.debug.callback(wandb_callback, metric, loss_info, adv, traj_batch, env_state, explained_var)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        # NOTE: instead of running the full jax.lax.scan here, return the
        # per-update step so the outer Python driver can checkpoint between
        # updates.
        return runner_state, _update_step

    return train


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--w", type=int, default=0,
                            help="1 = enable wandb logging (online), 0 = disabled (default)")
        parser.add_argument("--d", type=int, default=0)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--ent_coef", type=float, default=0.01)
        parser.add_argument("--ckpt_path", type=str, default="",
                            help="checkpoint folder name under ppo_checkpoints/ "
                                 "(empty = no checkpoints)")
        parser.add_argument("--params_only_checkpoint", action="store_true",
                            help="save only params/config. This avoids large "
                                 "solve_data checkpoint transfers on low-memory "
                                 "or native Windows runs.")
        parser.add_argument("--save_every", type=int, default=50,
                            help="save an Orbax checkpoint every N PPO updates")
        parser.add_argument("--resume_from", type=str, default="",
                            help="optional source ckpt folder to restore params "
                                 "from (separate from --ckpt_path). Useful for "
                                 "finetuning with a different config.")
        parser.add_argument("--resume_step", type=int, default=-1,
                            help="step to restore from when --resume_from is "
                                 "set; -1 = latest")
        parser.add_argument("--cycle_penalty", type=float, default=0.0,
                            help="per-step reward penalty when the new state "
                                 "has been visited earlier in this episode")
        parser.add_argument("--noop_penalty", type=float, default=0.0,
                            help="per-step reward penalty when the move did not "
                                 "change the state (length-overflow suppression)")
        parser.add_argument("--change_of_variables_moves", action="store_true",
                            help="enable stable change-of-variables moves")
        parser.add_argument("--ac45_moves", action="store_true",
                            help="enable AC4 add-generator and AC5 delete-generator moves")
        parser.add_argument("--stable_ac_moves", action="store_true",
                            help="deprecated alias for enabling both "
                                 "--change_of_variables_moves and --ac45_moves")
        parser.add_argument("--max_n_gen", type=int, default=2,
                            help="maximum generator capacity; PPO currently "
                                 "supports only 2 because the policy is a "
                                 "two-ring transformer")
        parser.add_argument("--seed", type=int, default=14,
                            help="PRNG seed for init + training rngs")
        return parser.parse_args()

    args = parse_args()

    change_of_variables_moves = args.change_of_variables_moves or args.stable_ac_moves
    ac45_moves = args.ac45_moves or args.stable_ac_moves

    config = {
        "LR": 5 * 5e-4,
        "NUM_ENVS": 1190 * 2,
        "NUM_STEPS": 96,
        "TOTAL_TIMESTEPS": 1e9,
        "UPDATE_EPOCHS": 3,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.999,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "gelu",
        "ENV_NAME": "AC-v0",
        "ANNEAL_LR": False,
        "DEBUG": args.d,
        "WANDB_MODE": "online" * args.w,
        "ENTITY": "",
        "PROJECT": "",
        "SEED": args.seed,
        "CYCLE_PENALTY": args.cycle_penalty,
        "NOOP_PENALTY": args.noop_penalty,
        "CHANGE_OF_VARIABLES_MOVES": change_of_variables_moves,
        "AC45_MOVES": ac45_moves,
        "STABLE_AC_MOVES": args.stable_ac_moves,
        "N_GEN": 2,
        "MAX_N_GEN": args.max_n_gen,
    }

    if config.get("WANDB_MODE", "disabled") == "online":
        import wandb
        wandb.init(
            entity=config["ENTITY"] or None,
            project=config["PROJECT"] or None,
            tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'purejaxrl_ppo_{config["ENV_NAME"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    # Build training pieces. make_train(...) returns a `train` closure that
    # when called returns (runner_state, per_update_step_fn).
    train_fn = make_train(config)
    runner_state, update_step_fn = train_fn(jax.random.PRNGKey(config["SEED"]))
    update_step_jit = jax.jit(update_step_fn)

    # Optional: load params from a different checkpoint folder before the
    # normal --ckpt_path resume logic runs. Lets us finetune from an existing
    # run with a new config (e.g., adding reward shaping) without clobbering
    # the source checkpoints.
    if args.resume_from:
        src_path_abs = os.path.join(os.getcwd(), "ppo_checkpoints", args.resume_from)
        src_options = ocp.CheckpointManagerOptions()
        src_manager = ocp.CheckpointManager(
            src_path_abs, options=src_options,
            item_names=("params", "solve_data", "config"),
        )
        src_step = src_manager.latest_step() if args.resume_step < 0 else args.resume_step
        if src_step is None:
            raise SystemExit(f"--resume_from {args.resume_from} has no checkpoints")
        print(f"Loading params from {src_path_abs} step {src_step}")
        dummy_params = runner_state[0].params
        restored = src_manager.restore(
            src_step,
            args=ocp.args.Composite(params=ocp.args.StandardRestore(dummy_params)),
        )
        train_state = runner_state[0].replace(params=restored.params)
        runner_state = (train_state, runner_state[1], runner_state[2], runner_state[3])
        src_manager.close()

    # Orbax setup
    ckpt_manager = None
    latest_step = 0
    if args.ckpt_path:
        ckpt_path_abs = os.path.join(os.getcwd(), "ppo_checkpoints", args.ckpt_path)
        print(f"Saving checkpoints to {ckpt_path_abs} every {args.save_every} updates")
        ckpt_item_names = (
            ("params", "config")
            if args.params_only_checkpoint
            else ("params", "solve_data", "config")
        )
        options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=args.save_every,
        )
        ckpt_manager = ocp.CheckpointManager(
            ckpt_path_abs,
            options=options,
            item_names=ckpt_item_names,
        )
        latest = ckpt_manager.latest_step()
        if latest is not None:
            print(f"Resuming params from checkpoint step {latest}")
            dummy_params = runner_state[0].params
            if args.params_only_checkpoint:
                restored = ckpt_manager.restore(
                    latest,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardRestore(dummy_params),
                        config=ocp.args.JsonRestore(config),
                    ),
                )
            else:
                dummy_solve_data = {
                    "solved_idx": runner_state[1].solved_idx,
                    "path_lengths": runner_state[1].path_lengths,
                    "best_paths": runner_state[1].best_paths,
                }
                restored = ckpt_manager.restore(
                    latest,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardRestore(dummy_params),
                        solve_data=ocp.args.StandardRestore(dummy_solve_data),
                        config=ocp.args.JsonRestore(config),
                    ),
                )
            train_state = runner_state[0].replace(params=restored.params)
            runner_state = (train_state, runner_state[1], runner_state[2], runner_state[3])
            latest_step = latest + 1
    else:
        print("No --ckpt_path given; skipping Orbax checkpointing.")

    print(f"Initial update step: {latest_step}, Final update step: {int(config['NUM_UPDATES']) - 1}")
    start_time = time.time()
    for u in range(latest_step, int(config["NUM_UPDATES"])):
        runner_state, _ = update_step_jit(runner_state, None)

        if ckpt_manager is not None:
            params = runner_state[0].params
            env_state = runner_state[1]
            # CheckpointManagerOptions(save_interval_steps=save_every) makes
            # save() a no-op except on cadence boundaries.
            if args.params_only_checkpoint:
                ckpt_manager.save(
                    u,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardSave(params),
                        config=ocp.args.JsonSave(config),
                    ),
                )
            else:
                solve_data = {
                    "solved_idx": env_state.solved_idx,
                    "path_lengths": env_state.path_lengths,
                    "best_paths": env_state.best_paths,
                }
                ckpt_manager.save(
                    u,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardSave(params),
                        solve_data=ocp.args.StandardSave(solve_data),
                        config=ocp.args.JsonSave(config),
                    ),
                )

        if config["DEBUG"]:
            env_state = runner_state[1]
            global_step = u * config["NUM_STEPS"] * config["NUM_ENVS"]
            num_solved = int(jnp.count_nonzero(env_state.solved_idx))
            now = time.time()
            sps = (config["NUM_STEPS"] * config["NUM_ENVS"]) / (now - start_time)
            print(f"update {u}, global step {global_step}, sps={sps // 1000}k, num_solved={num_solved}")
            start_time = now

    if ckpt_manager is not None:
        ckpt_manager.wait_until_finished()
