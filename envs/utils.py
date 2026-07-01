import numpy as np
import jax.numpy as jnp


def convert_relator_list_to_presentation(relators, max_relator_length, max_n_gen=None):
    """
    Converts a list of relators into a flat padded presentation.

    The output has ``max_n_gen`` relator slots, each of length
    ``max_relator_length``. When ``max_n_gen`` is omitted, it is the number of
    relators supplied.
    """
    if max_n_gen is None:
        max_n_gen = len(relators)
    assert len(relators) <= max_n_gen, "number of relators exceeds max_n_gen"
    padded = []
    for relator in relators:
        assert 0 not in relator, "relators must not be padded with zeros"
        assert len(relator) <= max_relator_length, (
            "max_relator_length must be greater than or equal to every relator length"
        )
        padded.extend(relator + [0] * (max_relator_length - len(relator)))

    for _ in range(max_n_gen - len(relators)):
        padded.extend([0] * max_relator_length)

    return np.array(padded, dtype=jnp.int8)


def convert_relators_to_presentation(relator1, relator2=None, max_relator_length=None,
                                     max_n_gen=None):
    """
    Backward-compatible wrapper for creating padded presentations.

    Existing callers may pass ``relator1, relator2, max_relator_length``.
    New callers may pass a list of relators as the first argument.
    """
    if relator2 is None and isinstance(relator1, list) and (
        len(relator1) == 0 or isinstance(relator1[0], list)
    ):
        assert max_relator_length is not None, "max_relator_length is required"
        return convert_relator_list_to_presentation(
            relator1, max_relator_length, max_n_gen
        )
    assert relator2 is not None, "relator2 is required for the two-relator form"
    assert max_relator_length is not None, "max_relator_length is required"
    assert isinstance(relator1, list) and isinstance(relator2, list), (
        f"got types {type(relator1)} for relator1 and {type(relator2)} for relator2"
    )
    return convert_relator_list_to_presentation(
        [relator1, relator2], max_relator_length, max_n_gen
    )


def change_presentation_shape(presentation, old_n_gen, new_max_length, max_n_gen=None):
    """
    Reformat a flat padded presentation to a new relator length and capacity.
    """
    if max_n_gen is None:
        max_n_gen = old_n_gen
    old_max_length = len(presentation) // old_n_gen
    relators = []
    for i in range(old_n_gen):
        start = i * old_max_length
        relator = presentation[start:start + old_max_length]
        word_length = np.count_nonzero(relator)
        relators.append(list(relator[:word_length]))
    return convert_relator_list_to_presentation(
        relators, new_max_length, max_n_gen=max_n_gen
    )


def change_max_relator_length_of_presentation(presentation, new_max_length):
    """
    Backward-compatible two-relator formatter.
    """
    return change_presentation_shape(presentation, 2, new_max_length, max_n_gen=2)


# --- Action (un)packing -----------------------------------------------------
# Paths are stored as a single packed integer per move (see
# wrappers.LogPathsProbsS.encode_action and the policy head in
# network.RelativeDualRingActorCritic), using L = max_length (24 in the
# training/beam scripts):
#
#     sample = (((k1 - 1) * L + (k2 + j) * (-1)**j) * 4) + (i * 2 + j)
#
# where the historical two-generator unpacked move is [i, j, k1, k2].
# For max_n_gen > 2, the first coordinate is a relator-pair branch.
# The functions below are the exact inverse used in ppo_ac_s.py / beam_search.py
# and also cover the generic environment path.


def num_substitution_branches(max_n_gen=2):
    """Number of substitution branch functions."""
    if max_n_gen == 2:
        return 2
    return max_n_gen * (max_n_gen - 1)


def num_substitution_actions(max_length=24, max_n_gen=2):
    """Number of packed substitution S-moves."""
    return num_substitution_branches(max_n_gen) * 2 * max_length * max_length


def num_change_of_variables_branches(max_n_gen=2):
    """Number of stable change-of-variables branch functions."""
    return max_n_gen * max_n_gen


def num_change_of_variables_actions(max_length=24, max_n_gen=2):
    """Number of optional stable change-of-variables moves."""
    return num_change_of_variables_branches(max_n_gen) * 2 * max_length * max_length


def num_stable_generator_actions(max_n_gen=2):
    """One AC4 add action plus one AC5 delete action per generator slot."""
    return 1 + max_n_gen


def change_of_variables_branch(remove_gen, iso_relator, max_n_gen=2):
    """Branch id for a stable change-of-variables action."""
    return (
        num_substitution_branches(max_n_gen)
        + int(remove_gen) * max_n_gen
        + int(iso_relator)
    )


def add_generator_branch(max_n_gen=2, change_of_variables_moves=False):
    """Branch id for AC4 add-generator."""
    offset = num_substitution_branches(max_n_gen)
    if change_of_variables_moves:
        offset += num_change_of_variables_branches(max_n_gen)
    return offset


def delete_generator_branch(max_n_gen=2, change_of_variables_moves=False):
    """Branch id for AC5 delete-generator."""
    return add_generator_branch(max_n_gen, change_of_variables_moves) + 1


def num_actions(max_length=24, change_of_variables_moves=False,
                ac45_moves=False, max_n_gen=2, stable_ac_moves=False):
    """Total packed action count for the configured environment."""
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    total = num_substitution_actions(max_length, max_n_gen)
    if change_of_variables_moves:
        total += num_change_of_variables_actions(max_length, max_n_gen)
    if ac45_moves:
        total += num_stable_generator_actions(max_n_gen)
    return total


def encode_action(action, max_length=24, change_of_variables_moves=False,
                  ac45_moves=False, max_n_gen=2, stable_ac_moves=False):
    """Pack one move into its stored integer index.

    Substitution moves use ``[branch, invert, k_target, k_source]``.
    For ``max_n_gen == 2`` the branch layout is the historical one.
    Change-of-variables moves start after the substitution branches when
    enabled. AC4/AC5 add/delete moves are appended after the enabled COV block.
    """
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    branch, j, k1, k2 = (int(action[0]), int(action[1]), int(action[2]), int(action[3]))
    L = max_length
    s_branches = num_substitution_branches(max_n_gen)
    s_count = num_substitution_actions(L, max_n_gen)
    cov_branches = num_change_of_variables_branches(max_n_gen)
    cov_count = (
        num_change_of_variables_actions(L, max_n_gen)
        if change_of_variables_moves else 0
    )

    if branch < s_branches:
        cell = (k1 - 1) * L + (k2 + j) * (-1) ** j
        return cell * (s_branches * 2) + (branch * 2 + j)

    if change_of_variables_moves and branch < s_branches + cov_branches:
        cov_branch = branch - s_branches
        cell = k1 * L + k2
        return s_count + cell * (cov_branches * 2) + (cov_branch * 2 + j)

    add_branch = add_generator_branch(max_n_gen, change_of_variables_moves)
    delete_branch = delete_generator_branch(max_n_gen, change_of_variables_moves)
    if ac45_moves and branch == add_branch:
        return s_count + cov_count
    if ac45_moves and branch == delete_branch:
        return s_count + cov_count + 1 + j
    raise ValueError(f"unknown action branch {branch}")


def decode_action(sample, max_length=24, change_of_variables_moves=False,
                  ac45_moves=False, max_n_gen=2, stable_ac_moves=False):
    """Decode one packed action index into an environment action vector."""
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    L = max_length
    a = int(sample)
    s_branches = num_substitution_branches(max_n_gen)
    s_count = num_substitution_actions(L, max_n_gen)
    cov_branches = num_change_of_variables_branches(max_n_gen)
    cov_count = (
        num_change_of_variables_actions(L, max_n_gen)
        if change_of_variables_moves else 0
    )
    if a < s_count:
        cell = a // (s_branches * 2)
        branch_j = a % (s_branches * 2)
        k1 = (cell // L) + 1
        k2_tmp = cell % L
        branch = branch_j // 2
        j = branch_j % 2
        k2 = k2_tmp * ((-1) ** j) - j
        return [int(branch), int(j), int(k1), int(k2)]

    if change_of_variables_moves and a < s_count + cov_count:
        offset = a - s_count
        cell = offset // (cov_branches * 2)
        channel = offset % (cov_branches * 2)
        z_start = cell // L
        z_len_code = cell % L
        cov_branch = channel // 2
        z_inverse = channel % 2
        return [int(s_branches + cov_branch), int(z_inverse), int(z_start), int(z_len_code)]

    if ac45_moves:
        gen_offset = a - s_count - cov_count
        add_branch = add_generator_branch(max_n_gen, change_of_variables_moves)
        delete_branch = delete_generator_branch(max_n_gen, change_of_variables_moves)
        if gen_offset == 0:
            return [int(add_branch), 0, 0, 0]
        return [int(delete_branch), int(gen_offset - 1), 0, 0]

    raise ValueError("packed action is outside substitution action range")


def encode_action_jax(action, max_length=24, change_of_variables_moves=False,
                      ac45_moves=False, max_n_gen=2, stable_ac_moves=False):
    """JAX version of ``encode_action`` supporting batched action arrays."""
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    action_branch = action[..., 0]
    action_j = action[..., 1]
    action_k1 = action[..., 2]
    action_k2 = action[..., 3]
    L = max_length
    s_branches = num_substitution_branches(max_n_gen)
    s_count = num_substitution_actions(L, max_n_gen)
    cov_branches = num_change_of_variables_branches(max_n_gen)
    cov_count = (
        num_change_of_variables_actions(L, max_n_gen)
        if change_of_variables_moves else 0
    )
    s_cell = (action_k1 - 1) * L + (action_k2 + action_j) * (-1) ** action_j
    s_encoded = s_cell * (s_branches * 2) + (action_branch * 2 + action_j)
    if not (change_of_variables_moves or ac45_moves):
        return s_encoded

    cov_start_branch = s_branches
    add_branch = add_generator_branch(max_n_gen, change_of_variables_moves)
    cov_branch = action_branch - cov_start_branch
    cov_cell = action_k1 * L + action_k2
    cov_encoded = s_count + cov_cell * (cov_branches * 2) + (cov_branch * 2 + action_j)
    add_encoded = jnp.full_like(s_encoded, s_count + cov_count)
    delete_encoded = s_count + cov_count + 1 + action_j
    if change_of_variables_moves:
        is_cov_branch = action_branch < add_branch
    else:
        is_cov_branch = jnp.zeros_like(action_branch, dtype=jnp.bool_)
    stable_encoded = jnp.where(
        is_cov_branch,
        cov_encoded,
        jnp.where(action_branch == add_branch, add_encoded, delete_encoded),
    )
    return jnp.where(action_branch < s_branches, s_encoded, stable_encoded)


def decode_action_jax(sample, max_length=24, change_of_variables_moves=False,
                      ac45_moves=False, max_n_gen=2, stable_ac_moves=False):
    """JAX version of ``decode_action`` supporting scalar or batched indices."""
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    sample = jnp.asarray(sample, dtype=jnp.int32)
    L = max_length
    s_branches = num_substitution_branches(max_n_gen)
    s_count = num_substitution_actions(L, max_n_gen)
    cov_branches = num_change_of_variables_branches(max_n_gen)
    cov_count = (
        num_change_of_variables_actions(L, max_n_gen)
        if change_of_variables_moves else 0
    )

    s_cell = sample // (s_branches * 2)
    branch_j = sample % (s_branches * 2)
    action_k1 = (s_cell // L) + 1
    action_k2_tmp = s_cell % L
    action_i = branch_j // 2
    action_j = branch_j % 2
    action_k2 = action_k2_tmp * (-1) ** action_j - action_j
    s_action = jnp.stack([action_i, action_j, action_k1, action_k2], axis=-1)
    if not (change_of_variables_moves or ac45_moves):
        return s_action

    offset = sample - s_count
    cell = offset // (cov_branches * 2)
    channel = offset % (cov_branches * 2)
    z_start = cell // L
    z_len_code = cell % L
    cov_branch = channel // 2
    z_inverse = channel % 2
    cov_action = jnp.stack([s_branches + cov_branch, z_inverse, z_start, z_len_code], axis=-1)

    gen_offset = sample - s_count - cov_count
    add_branch = add_generator_branch(max_n_gen, change_of_variables_moves)
    delete_branch = delete_generator_branch(max_n_gen, change_of_variables_moves)
    add_action = jnp.stack([
        jnp.full_like(sample, add_branch),
        jnp.zeros_like(sample),
        jnp.zeros_like(sample),
        jnp.zeros_like(sample),
    ], axis=-1)
    delete_action = jnp.stack([
        jnp.full_like(sample, delete_branch),
        gen_offset - 1,
        jnp.zeros_like(sample),
        jnp.zeros_like(sample),
    ], axis=-1)
    if change_of_variables_moves:
        is_cov_sample = sample < s_count + cov_count
    else:
        is_cov_sample = jnp.zeros_like(sample, dtype=jnp.bool_)
    stable_action = jnp.where(
        is_cov_sample[..., None],
        cov_action,
        jnp.where((gen_offset == 0)[..., None], add_action, delete_action),
    )
    return jnp.where((sample < s_count)[..., None], s_action, stable_action)


def decode_path(path, max_length=24, pad_value=-1,
                change_of_variables_moves=False, ac45_moves=False,
                max_n_gen=2, stable_ac_moves=False):
    """Decode a stored path of packed action indices into action vectors.

    `path` is an iterable of packed integers (e.g. a row of
    env_state.best_paths). Padding entries equal to `pad_value` are dropped, so
    a fixed-width padded row decodes to just its real moves.
    """
    return [
        decode_action(
            a, max_length, change_of_variables_moves, ac45_moves,
            max_n_gen, stable_ac_moves,
        )
        for a in path if int(a) != pad_value
    ]


# --- Path validation --------------------------------------------------------
def replay_packed_path(env, idx, packed_path, max_length=24,
                       change_of_variables_moves=False, ac45_moves=False,
                       max_n_gen=2, stable_ac_moves=False):
    """Replay one stored (packed) path in an ACS env from initial state `idx`.

    `packed_path` is a sequence of packed action indices (e.g. a row of
    env_state.best_paths), -1-padded. Returns (terminated, n_steps, final_x):
      terminated : True iff a trivial presentation was reached
      n_steps    : number of moves applied before stopping
      final_x    : the final presentation as a numpy array
    """
    import jax
    import jax.numpy as jnp

    params = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params,
                             idx=jnp.int32(int(idx)),
                             sample=jnp.bool_(False))
    terminated = False
    n_steps = 0
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    change_of_variables_moves = (
        change_of_variables_moves
        or getattr(env, "change_of_variables_moves", False)
    )
    ac45_moves = ac45_moves or getattr(env, "ac45_moves", False)
    max_n_gen = getattr(env.default_params, "max_n_gen", max_n_gen)
    for move in decode_path(packed_path, max_length=max_length,
                            change_of_variables_moves=change_of_variables_moves,
                            ac45_moves=ac45_moves,
                            max_n_gen=max_n_gen):
        _, state, _, _, info = env.step_env(
            key, state, jnp.asarray(move, dtype=jnp.int32), params
        )
        n_steps += 1
        terminated = bool(info["terminated"])
        if terminated:
            break
    return terminated, n_steps, np.asarray(state.x)


def check_paths(solved_idx, path_lengths, best_paths, initial_states_file,
                max_length=24, max_paths=None, verbose=True,
                change_of_variables_moves=False, ac45_moves=False,
                n_gen=2, max_n_gen=None, stable_ac_moves=False):
    """Validate stored substitution paths by replaying them in the ACS env.

    For every index flagged in `solved_idx`, replays best_paths[idx] starting
    from init_states[idx] of `initial_states_file` and checks that it reaches
    the trivial presentation in exactly path_lengths[idx] moves.

    `solved_idx`, `path_lengths`, `best_paths` are the arrays held in
    env_state.solved_idx / .path_lengths / .best_paths (and saved in an Orbax
    checkpoint's solve_data). best_paths holds packed action indices padded
    with -1.

    `max_paths` (optional) caps how many solved paths are checked (the first
    `max_paths` solved indices); None checks all of them.

    Returns the list of indices that FAILED to validate (empty => all good).
    """
    # Lazy import: ac_s imports this module, so importing it at top level would
    # create a circular import.
    from envs.ac_s import ACS

    solved_idx = np.asarray(solved_idx)
    path_lengths = np.asarray(path_lengths)
    best_paths = np.asarray(best_paths)

    # Size max_steps to the stored path width so episodes never truncate before
    # a path finishes; truncation would not affect the terminal check anyway.
    max_steps = int(best_paths.shape[1])
    if max_n_gen is None:
        max_n_gen = n_gen
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    env = ACS(n_gen=n_gen, max_length=max_length, max_steps_in_episode=max_steps,
              is_reward_sparse=False, initial_states_file=initial_states_file,
              change_of_variables_moves=change_of_variables_moves,
              ac45_moves=ac45_moves, max_n_gen=max_n_gen)

    failures = []
    solved_indices = np.nonzero(solved_idx)[0]
    if max_paths is not None:
        solved_indices = solved_indices[:max_paths]
    for idx in solved_indices:
        terminated, n_steps, _ = replay_packed_path(
            env, idx, best_paths[idx], max_length=max_length,
            change_of_variables_moves=change_of_variables_moves,
            ac45_moves=ac45_moves, max_n_gen=max_n_gen
        )
        expected = int(path_lengths[idx])
        if not (terminated and n_steps == expected):
            failures.append(int(idx))
            if verbose:
                print(f"[idx {idx}] INVALID: terminated={terminated}, "
                      f"replay_steps={n_steps}, stored_length={expected}")
    if verbose:
        n = len(solved_indices)
        print(f"checked {n} solved paths: {n - len(failures)} valid, "
              f"{len(failures)} invalid")
    return failures
