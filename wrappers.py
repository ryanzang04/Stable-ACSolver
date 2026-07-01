import jax
from jax import lax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from envs.utils import encode_action_jax


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None, idx: int = 0, sample: bool = False, probs: Optional[chex.Array] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params, idx, sample, probs)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
        probs: Optional[chex.Array] = None
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params, probs
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


@struct.dataclass
class LogPathsProbsGlobalState:
    solves: jnp.ndarray
    attempts: jnp.ndarray
    probs: jnp.ndarray
    solved_idx: jnp.ndarray
    env_state: environment.EnvState
    path_lengths: jnp.ndarray
    best_paths: jnp.ndarray
    current_actions: jnp.ndarray


class LogPathsProbsS(GymnaxWrapper):
    """Vectorized environment for ACS. Keeps track of solved states, the best
    (shortest) solving path found per initial state, and an adaptive sampling
    distribution over the initial states (harder/less-attempted states get
    sampled more)."""

    def __init__(self, env, num_envs):
        super().__init__(env)
        self.vmap_reset = jax.vmap(self._env.reset, in_axes=(0, None, 0, 0, None))
        self.vmap_step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None, None))
        self.num_envs = num_envs
        self.beta = 5
        self.alpha = 1 #1

        # NOTE: can possibly be confusing to have different placeholders but leaving it as it is for now
        self.path_length_placeholder = jnp.iinfo(jnp.int32).max
        self.best_paths_placeholder = -1
        self.current_actions_placeholder = -1

    def reset(self, key, params=None, idx=None, sample=False, probs=None):
        obsv, env_state = self.vmap_reset(key, params, idx,  sample, probs)

        # In ppo_ac.py, we only call reset() once and then never again.
        state = LogPathsProbsGlobalState(
            solves=jnp.zeros(len(self._env.init_states), dtype=jnp.int32),
            attempts=jnp.zeros(len(self._env.init_states), dtype=jnp.int32),
            probs=jnp.ones(len(self._env.init_states), dtype=jnp.float32) / len(self._env.init_states),
            solved_idx=jnp.zeros(len(self._env.init_states), dtype=jnp.bool_),
            env_state=env_state,
            path_lengths = jnp.full(len(self.init_states), self.path_length_placeholder, dtype=jnp.int32),
            best_paths = jnp.full((len(self.init_states), self.params.max_steps_in_episode), self.best_paths_placeholder, dtype=jnp.int32),
            current_actions = jnp.full((self.num_envs, self.params.max_steps_in_episode), self.current_actions_placeholder, dtype=jnp.int32), # (N, E) <-- prepared for vmapped actions
        )
        return obsv, state

    @staticmethod
    def _propagate_min_value(current_idx_batch, lengths_batch, paths_batch, terminated_flags_batch):
        """
        Propagates the minimum length path and associated info for each unique initial state index.
        All inputs are per-environment (batched, shape (num_envs, ...)).
        Outputs are also per-environment, but consistent for envs mapping to the same init_state_idx.

        Example:
        Let N = config["NUM_ENVS"] = 10 parallel envs.
        current_idx_batch = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        lengths_batch = [l, 33, 40, l, l, l, l, 40, l, l] where l is the large number jnp.iinfo(jnp.int32).max
        ....

        This method computes:
        sorted_indices = (0, 5, 1, 6, 7, 2, 3, 8, 4, 9)
        idx_sorted = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4,)
        paths_sorted = ... # (N, M) for M=max_lengths_in_episode
        lengths_sorted = [l, l, 33, l,m 40, 40, l, l, l, l]
        terminated_sorted = (F, F, T, F, T, T, F, F, F, F)
        is_new_group = (T, F, T, F, T, F, T, F, T, F) # start of new group
        propagated_paths = ... # (N, M)
        propagated_lengths = (l, l, 33, 33, 40, 40, l, l, l, l)
        propagated_terminated = (F, F, T, T, T, T, F, F, F, F)
        inv_sort_indices = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
        final_paths = ... # (N, M)
        final_lengths = [l, 33, 40, l, l, l, 33, 40, l, l]
        """
        # Sort by (init_state_idx, length) to find the best path for each init_state_idx
        # jnp.lexsort sorts by the last key first. To sort by current_idx_batch then lengths_batch:
        sorted_indices = jnp.lexsort((lengths_batch, current_idx_batch))

        idx_sorted = current_idx_batch[sorted_indices] # (N, )
        paths_sorted = paths_batch[sorted_indices] # (N, M) where M is the max_length_in_episode = 200
        lengths_sorted = lengths_batch[sorted_indices] # (N, )
        terminated_sorted = terminated_flags_batch[sorted_indices] # (N, )

        # Identify the start of each new group of init_state_idx
        is_new_group = jnp.concatenate([jnp.array([True]), idx_sorted[1:] != idx_sorted[:-1]]) # (N, )

        def scan_fn(carry, x):
            val, new_group_flag = x
            # If it's a new group, take the current item's value (which is the best for this group due to sorting).
            # Otherwise, keep carrying forward the value from the start of the current group.
            out = lax.select(new_group_flag, val, carry)
            return out, out

        # Propagate the first value (best path's attributes) within each group
        # Initial carry for scan is the first element of the sorted array.
        # Requires num_envs > 0.
        _, propagated_paths = lax.scan(scan_fn, paths_sorted[0], (paths_sorted, is_new_group))
        _, propagated_lengths = lax.scan(scan_fn, lengths_sorted[0], (lengths_sorted, is_new_group))
        _, propagated_terminated = lax.scan(scan_fn, terminated_sorted[0], (terminated_sorted, is_new_group))

        # Unsort to restore original order of environments
        inv_sort_indices = jnp.zeros_like(sorted_indices).at[sorted_indices].set(jnp.arange(len(sorted_indices)))

        final_lengths = propagated_lengths[inv_sort_indices]
        final_paths = propagated_paths[inv_sort_indices]
        final_terminated = propagated_terminated[inv_sort_indices]

        return final_lengths, final_paths, final_terminated

    def _compute_probs(self, state, terminated, done, current_idx):
        # where terminated, add 1 to solves
        new_solves = state.solves.at[current_idx].add(terminated)
        # where done, add 1 to attempts
        new_attempts = state.attempts.at[current_idx].add(done)
        # update probs so that probs = (solves + 1) / (attempts + 1)
        success_rate = new_solves / (new_attempts + self.beta)

        # high success rate and often picked both make probability go down
        raw_probs = (1-success_rate)**(self.alpha) / (1+new_attempts)**0.5
        new_probs = raw_probs / jnp.sum(raw_probs)

        state = state.replace(
            probs=new_probs,
            solves=new_solves,
            attempts=new_attempts,
        )
        return state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self.vmap_step(
            key, state.env_state, action, params, state.probs
        )

        current_idx = info['idx'] # (NUM_ENVS, )
        terminated = info['terminated'] # (NUM_ENVS, )
        episode_length = info['episode_length'] # (NUM_ENVS, ) # state.time coming from AC.step.
        encoded_action = encode_action_jax(action, self.params.max_length,
                                           self.change_of_variables_moves,
                                           self.ac45_moves,
                                           max_n_gen=self.params.max_n_gen)
        # always update current actions
        new_current_actions = state.current_actions.at[
            jnp.arange(self.num_envs,), episode_length-1
            ].set(encoded_action) #(N, E)

        # check whether any new best paths have been found
        new_best_path_found = terminated & (
            (episode_length < state.path_lengths[current_idx]) #| # (N, ) | (N, )
            ) # (NUM_ENVS, )

        def update_best_paths(current_idx, new_best_path_found, episode_length, state, new_current_actions):
            #lengths is the lengths of best path, it can be computed on the fly, but maybe this is equally efficient
            new_lengths = jax.lax.select(new_best_path_found,
                                        episode_length,
                                        state.path_lengths[current_idx]) # (N, )

            new_best_path_found_expanded = jnp.broadcast_to(
                new_best_path_found[..., None],
                new_best_path_found.shape + (self.params.max_steps_in_episode,)
            ) # Broadcast new_best_path_found with shape = (N,) to (N, E)

            new_paths = jax.lax.select(new_best_path_found_expanded,
                                        new_current_actions,
                                        state.best_paths[current_idx])

            done_expanded = jnp.broadcast_to(
                done[..., None],
                done.shape + (self.params.max_steps_in_episode,)
            ) # Brodcast done with shape = (N,) to done_expanded with shape = (N, E)

            new_current_actions = jax.lax.select(done_expanded,
                                                jnp.full((self.num_envs, self.params.max_steps_in_episode), -1, dtype=jnp.int32),
                                                new_current_actions)

            new_terminated = terminated | state.solved_idx[current_idx]

            new_lengths, new_paths, new_terminated = self._propagate_min_value(current_idx, new_lengths, new_paths, new_terminated)

            new_state = self._compute_probs(state, terminated, done, current_idx)

            state = LogPathsProbsGlobalState(
                solves = new_state.solves,
                attempts = new_state.attempts,
                probs = new_state.probs,
                solved_idx=state.solved_idx.at[current_idx].set(new_terminated),
                path_lengths=state.path_lengths.at[current_idx].set(new_lengths),
                best_paths=state.best_paths.at[current_idx].set(new_paths),
                current_actions=new_current_actions,
                env_state=env_state,
            )
            return state

        def no_update_best_paths(current_idx, new_best_path_found, episode_length, state, new_current_actions):
            # If no new best path found, just return the state as with the actions updated
            done_expanded = jnp.broadcast_to(
                done[..., None],
                done.shape + (self.params.max_steps_in_episode,)
            ) # Brodcast done with shape = (N,) to done_expanded with shape = (N, E)

            new_current_actions = jax.lax.select(done_expanded,
                                                jnp.full((self.num_envs, self.params.max_steps_in_episode), -1, dtype=jnp.int32),
                                                new_current_actions)

            # where terminated, add 1 to solves
            new_state = self._compute_probs(state, terminated, done, current_idx)

            return LogPathsProbsGlobalState(
                solves = new_state.solves,
                attempts = new_state.attempts,
                probs = new_state.probs,
                solved_idx=state.solved_idx,
                path_lengths=state.path_lengths,
                best_paths=state.best_paths,
                current_actions=new_current_actions,
                env_state=env_state,
            )

        state = jax.lax.cond(
            jnp.any(new_best_path_found),
            update_best_paths,
            no_update_best_paths,
            current_idx,
            new_best_path_found,
            episode_length,
            state,
            new_current_actions,
        )

        return obs, state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None, idx=None, sample=False, probs=None):
        obs, state = self._env.reset(key, params, idx, sample, probs)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None, probs=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params, probs
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
