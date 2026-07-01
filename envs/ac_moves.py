import jax
import jax.lax as lax
import jax.numpy as jnp
from functools import partial
from envs.environment import EnvParams
from envs.utils import num_substitution_branches, num_change_of_variables_branches


# Original presentation-slot helpers used by the substitution moves.
# These operate on the full flat presentation `x`.
def _invert(i: int, params: EnvParams, x: jnp.ndarray):
    """Inverts the ith relator."""
    max_length = params.max_length
    ith_relator = lax.dynamic_slice(x, (i*max_length,), (max_length,))
    ith_relator_inverted = jnp.where(ith_relator != 0, -_reverse_nonzero(ith_relator), jnp.int8(0))
    return jax.lax.dynamic_update_slice(x, ith_relator_inverted, (i*max_length,)) # type: ignore


# Note that if the result of a concatenation is longer than the maximum length,
# it is not stored in the relators. This is a limitation of the current implementation.
# If the result is longer than the maximum length, it is simply discarded. I don't know how to mask this action.
# Simplest thing to do is to add a negative reward if the result does not change.
# Probably also add a negative reward if there are zero simplifications.
def _concatenate(i: int, j: int, params: EnvParams, x: jnp.ndarray):
    """Concatenates the ith and jth relators."""
    max_length = params.max_length
    ith_relator = lax.dynamic_slice(x, (i*max_length,), (max_length,))
    jth_relator = lax.dynamic_slice(x, (j*max_length,), (max_length,))

    # r_i = a c, r_j = C b a #
    ith_relator_reversed = _reverse_nonzero(ith_relator) # r_i (reversed) = c a
    mask = (jth_relator == - ith_relator_reversed) # mask = (C b a == C a) = T F F

    num_cancel = jnp.argmin(mask) # 1 <-- = the number of elements that must cancel

    ith_len = jnp.count_nonzero(ith_relator)
    jth_len = jnp.count_nonzero(jth_relator)
    new_size = ith_len + jth_len - 2 * num_cancel

    def do_nothing(x, ith_relator, jth_relator, ith_len, num_cancel, new_size):
        return x

    def update_x(x, ith_relator, jth_relator, ith_len, num_cancel, new_size):

        # mask1 and mask2 specify indices of updated_ith_relator
        # where elements of ith_relator and jth_relator should be copied
        positions = jnp.arange(max_length, dtype=jnp.int8)
        mask1 = jnp.zeros_like(positions, dtype=jnp.bool_)
        mask2 = jnp.zeros_like(positions, dtype=jnp.bool_)

        # ith_len = 2, num_cancel = 1, so at position = 0, set mask1 = 1.
        # new_size = 2 + 3 - 2 = 3, ith_len = 2, num_cancel = 1
        # so for positions >= 1 and positions < 3, i.e. positions = [1, 2], set mask2=1.
        mask1 = jnp.where(positions < ith_len - num_cancel, 1, 0)
        mask2 = jnp.where(jnp.logical_and(positions >= ith_len - num_cancel, positions < new_size), 1, 0)

        # rotate jth_relator by ith_len - 2 * num_cancel = 2 - 2 = 0 elements.
        # where mask2=True, i.e. positions = [1, 2], place the first and the second element,
        # i.e. b a (skipping C as C is to be cancelled.)
        # where mask1=True, place
        updated_ith_relator = jnp.zeros_like(ith_relator)
        updated_ith_relator = jnp.where(
            mask2,
            jnp.roll(jth_relator, ith_len - 2 * num_cancel),
            jnp.where(mask1, ith_relator, 0) # type: ignore
        )

        out = jax.lax.dynamic_update_slice(x, updated_ith_relator, (i*max_length,))
        return out

    out = jax.lax.cond(
        new_size > max_length,
        do_nothing,
        update_x,
        x, ith_relator, jth_relator, ith_len, num_cancel, new_size,
    )

    out = cyclic_reduce(i, params, out)

    return out


def cyclic_reduce(i: int, params: EnvParams, x: jnp.ndarray):
    """only need to reduce one relator; the one that was just modified: labelled by i."""
    max_length = params.max_length

    # C a b c
    ith_relator = lax.dynamic_slice(x, (i*max_length,), (max_length,)) # C a b c
    ith_relator_reversed = _reverse_nonzero(ith_relator) # c b a C

    ith_len = jnp.count_nonzero(ith_relator)

    #
    mask = (ith_relator == - ith_relator_reversed) # C a b c == C B A c --> T F F T

    # get index of the first F, eq. the total number of letters on each end to cancel
    indices = jnp.arange(max_length)
    num_cancel = jnp.min(jnp.where(~mask, indices, max_length))

    # We don't have to worry about length.
    # just copy [num_cancel: ith_len - num_cancel] = [1: 4-1] = [1: 3] = a b
    # at the beginning of the updated_ith_relator.
    rolled_indices = (indices + num_cancel) % max_length # [-1, 0, 1, 2]
    updated_ith_relator = jnp.where(
        indices >= ith_len - 2 * num_cancel, # 2, 3, ...
        jnp.zeros_like(ith_relator),
        ith_relator[rolled_indices]
    )

    out = jax.lax.dynamic_update_slice(x, updated_ith_relator, (i*max_length,))

    return out


def _reverse_nonzero(arr: jnp.ndarray):
    """Reverses the nonzero elements of the array."""
    nonzero_mask = arr != 0

    positions = jnp.arange(arr.shape[0])
    # Calculate new positions for non-zero elements
    # If the first 3 elements are non-zero in a length-5 array,
    # this maps [0,1,2,3,4] to [2,1,0,3,4]
    nonzero_count = jnp.sum(nonzero_mask)
    new_positions = jnp.where(
        nonzero_mask,
        nonzero_count - 1 - positions,
        positions
    )

    # Use the positions to create the reversed array
    reversed_arr = jnp.zeros_like(arr)
    reversed_arr = reversed_arr.at[new_positions].set(arr)

    return reversed_arr.astype(arr.dtype)


# Generic padded-word helpers used by change-of-variables and AC5.
# These operate on standalone words or generator labels, not whole moves.
def _inverse_word(arr: jnp.ndarray):
    """Returns the inverse of a padded word."""
    return -_reverse_nonzero(arr)


def _concat_fixed(*words: jnp.ndarray) -> jnp.ndarray:
    """Concatenate padded words into a fixed-size buffer of their total size."""
    out = jnp.zeros((sum(word.shape[0] for word in words),), dtype=words[0].dtype)
    offset = 0
    for word in words:
        length = jnp.count_nonzero(word)
        positions = jnp.arange(word.shape[0])
        idx = offset + positions
        out = out.at[idx].set(jnp.where(positions < length, word, jnp.array(0, dtype=word.dtype)))
        offset += word.shape[0]
    return out


def _free_reduce_word(word: jnp.ndarray) -> jnp.ndarray:
    """Free-reduce a padded word with a stack, preserving the input shape."""
    capacity = word.shape[0]

    def step(carry, token):
        out, size = carry
        top_idx = jnp.maximum(size - 1, 0)
        top = out[top_idx]
        is_letter = token != 0
        cancels = (size > 0) & is_letter & (top == -token)

        pop_out = out.at[top_idx].set(jnp.array(0, dtype=word.dtype))
        push_idx = jnp.minimum(size, capacity - 1)
        push_out = out.at[push_idx].set(token)

        new_out = jax.lax.select(cancels, pop_out, push_out)
        new_out = jax.lax.select(is_letter, new_out, out)
        new_size = jnp.where(is_letter, size + jnp.where(cancels, -1, 1), size)
        return (new_out, new_size), None

    init = (jnp.zeros_like(word), jnp.array(0, dtype=jnp.int32))
    (out, _), _ = lax.scan(step, init, word)
    return out


def _cyclic_reduce_word(word: jnp.ndarray) -> jnp.ndarray:
    """Cyclically reduce a free-reduced padded word."""
    capacity = word.shape[0]
    length = jnp.count_nonzero(word)
    reversed_word = _reverse_nonzero(word)
    mask = word == -reversed_word
    indices = jnp.arange(capacity)
    num_cancel = jnp.min(jnp.where(~mask, indices, capacity))
    rolled_indices = (indices + num_cancel) % capacity
    reduced_len = length - 2 * num_cancel
    return jnp.where(indices >= reduced_len, jnp.zeros_like(word), word[rolled_indices])


def _reduce_to_relator(word: jnp.ndarray, max_length: int):
    """Free/cyclic reduce `word` and copy it into one relator slot."""
    reduced = _cyclic_reduce_word(_free_reduce_word(word))
    reduced_len = jnp.count_nonzero(reduced)
    positions = jnp.arange(max_length)
    relator = jnp.take(reduced, positions, mode="clip")
    relator = jnp.where(positions < reduced_len, relator, jnp.zeros_like(relator))
    return relator.astype(word.dtype), reduced_len > max_length


def _slice_relator(x: jnp.ndarray, idx: jnp.ndarray, max_length: int) -> jnp.ndarray:
    """Slice a relator slot by dynamic index."""
    return lax.dynamic_slice(x, (idx * max_length,), (max_length,))


def _update_relator(x: jnp.ndarray, idx: jnp.ndarray, relator: jnp.ndarray,
                    max_length: int) -> jnp.ndarray:
    """Update a relator slot by dynamic index."""
    return lax.dynamic_update_slice(x, relator, (idx * max_length,))


def _renumber_after_delete(x: jnp.ndarray, deleted_code: jnp.ndarray) -> jnp.ndarray:
    """Shift generator labels above deleted_code down by one."""
    return jnp.where(
        x > deleted_code,
        x - 1,
        jnp.where(x < -deleted_code, x + 1, x),
    ).astype(x.dtype)


def _substitute_generator(word: jnp.ndarray, old_code: jnp.ndarray, replacement: jnp.ndarray):
    """Replace old_code in `word` by `replacement` and -old_code by inverse."""
    word_len = word.shape[0]
    repl_len_static = replacement.shape[0]
    capacity = word_len * repl_len_static
    inv_replacement = _inverse_word(replacement)
    replacement_len = jnp.count_nonzero(replacement)
    positions = jnp.arange(repl_len_static)

    def step(carry, token):
        out, cursor = carry
        singleton = jnp.zeros_like(replacement).at[0].set(token)
        use_repl = token == old_code
        use_inv = token == -old_code
        is_letter = token != 0
        segment = jnp.where(use_repl, replacement,
                            jnp.where(use_inv, inv_replacement, singleton))
        segment_len = jnp.where(use_repl | use_inv, replacement_len,
                                jnp.where(is_letter, 1, 0))
        idx = cursor + positions
        values = jnp.where(positions < segment_len, segment, jnp.zeros_like(segment))
        out = out.at[idx].set(values, mode="drop")
        cursor = cursor + segment_len
        return (out, cursor), None

    init = (jnp.zeros((capacity,), dtype=word.dtype), jnp.array(0, dtype=jnp.int32))
    (out, _), _ = lax.scan(step, init, word)
    return out


# Change-of-variables move: replace one old generator slot by a new variable z.
def _change_of_variables(remove_gen: int, iso_relator: int, params, x: jnp.ndarray,
                         active_n_gen: jnp.ndarray, z_args: jnp.ndarray):
    """Stable change-of-variables supermove.

    The action chooses a cyclic subword W of one relator, introduces z = W
    (or z = W^{-1}), uses the complementary part of that relator to solve for
    one old generator, removes that old generator, and reuses its numeric slot
    for z. Invalid or overflowing choices leave the state unchanged.
    """
    # Decode the action arguments. The length is stored as z_len - 1 so that
    # all valid lengths fit naturally in the action range [0, max_length).
    max_length = params.max_length
    z_inverse, z_start, z_len_code = z_args[0], z_args[1], z_args[2]
    z_len = z_len_code + 1
    old_code = jnp.array(remove_gen + 1, dtype=x.dtype)

    # Extract the relator used to isolate the old generator.
    iso_idx = jnp.array(iso_relator, dtype=jnp.int32)
    iso = _slice_relator(x, iso_idx, max_length)
    iso_len = jnp.count_nonzero(iso)
    safe_iso_len = jnp.maximum(iso_len, 1)

    # Rotate the relator so the selected cyclic subword W starts at index 0.
    # `window` is W, and `complement` is the rest of the cyclic relator.
    positions = jnp.arange(max_length)
    rotated_idx = (z_start + positions) % safe_iso_len
    rotated = jnp.where(positions < iso_len, jnp.take(iso, rotated_idx), jnp.zeros_like(iso))
    window = jnp.where(positions < z_len, rotated, jnp.zeros_like(rotated))
    complement_len = iso_len - z_len
    complement_idx = (z_len + positions) % max_length
    complement = jnp.where(positions < complement_len, rotated[complement_idx],
                           jnp.zeros_like(rotated))

    # Define the new variable z as either W or W^{-1}.
    z_word = lax.cond(
        z_inverse == 1,
        _inverse_word,
        lambda word: word,
        window,
    )

    # The complement must contain the old generator exactly once, so we can
    # solve the isolating relator for that generator.
    old_occurrences = jnp.abs(complement) == old_code
    old_count = jnp.sum(old_occurrences)
    old_pos = jnp.argmax(old_occurrences)
    old_token = complement[old_pos]

    # Split complement = before * old_token * after.
    before = jnp.where(positions < old_pos, complement, jnp.zeros_like(complement))
    after_idx = old_pos + 1 + positions
    after = jnp.where(positions < complement_len - old_pos - 1,
                      jnp.take(complement, after_idx, mode="clip"),
                      jnp.zeros_like(complement))

    # Solve the equation for the old generator. The formula depends on whether
    # the old generator appears with positive or negative sign.
    z_rhs = jnp.where(z_inverse == 1, old_code, -old_code)
    z_rhs_word = jnp.zeros_like(complement).at[0].set(z_rhs)
    z_lhs_word = jnp.zeros_like(complement).at[0].set(-z_rhs)
    solved_positive = _concat_fixed(_inverse_word(before), z_rhs_word, _inverse_word(after))
    solved_negative = _concat_fixed(after, z_lhs_word, before)
    solved_expr_long = jnp.where(old_token == old_code, solved_positive, solved_negative)
    solved_expr, expr_overflow = _reduce_to_relator(solved_expr_long, max_length)

    # Build the defining relator for z. Since z reuses old_code's numeric slot,
    # this is the relator z * z_word^{-1}, after substituting out the old code.
    definition_tail = _inverse_word(z_word)
    definition_tail_substituted = _substitute_generator(definition_tail, old_code, solved_expr)
    z_prefix = jnp.zeros((1,), dtype=x.dtype).at[0].set(old_code)
    definition_word = jnp.concatenate([z_prefix, definition_tail_substituted])
    definition_relator, definition_overflow = _reduce_to_relator(definition_word, max_length)

    # Substitute the solved expression into every other active relator.
    def body(idx, carry):
        out, overflow = carry
        relator = _slice_relator(x, idx, max_length)
        substituted = _substitute_generator(relator, old_code, solved_expr)
        new_relator, rel_overflow = _reduce_to_relator(substituted, max_length)
        active_other = (idx < active_n_gen) & (idx != iso_idx)
        new_relator = jnp.where(active_other, new_relator, relator)
        out = _update_relator(out, idx, new_relator, max_length)
        overflow = overflow | (active_other & rel_overflow)
        return out, overflow

    updated, substitution_overflow = lax.fori_loop(
        0, params.max_n_gen, body, (x, jnp.array(False))
    )
    # Replace the isolating relator by the defining relator for z, then rotate
    # it to the canonical cyclic representative used elsewhere in the env.
    updated = _update_relator(updated, iso_idx, definition_relator, max_length)
    updated = rotate_relator_k(
        iso_relator, booth_lex_min_rotation_masked(definition_relator),
        params, updated,
    )

    # Reject invalid choices and any move that would overflow a relator slot.
    valid = (
        (remove_gen < active_n_gen)
        & (iso_relator < active_n_gen)
        & (iso_len > 1)
        & (z_start < iso_len)
        & (z_len < iso_len)
        & (old_count == 1)
        & (~expr_overflow)
        & (~substitution_overflow)
        & (~definition_overflow)
    )
    # Accepted moves return the updated presentation. Rejected moves are no-ops.
    return lax.cond(
        valid,
        lambda _: (updated, active_n_gen),
        lambda _: (x, active_n_gen),
        operand=None,
    )


# Stable AC generator-count moves. AC4 adds a trivial generator/relator;
# AC5 removes a generator isolated by a trivial relator.
def _add_generator(params, x: jnp.ndarray, active_n_gen: jnp.ndarray,
                   args: jnp.ndarray):
    """AC4: append x_{n+1} as a new trivial relator when capacity allows."""
    max_length = params.max_length
    valid = active_n_gen < params.max_n_gen
    relator = jnp.zeros((max_length,), dtype=x.dtype).at[0].set(
        (active_n_gen + 1).astype(x.dtype)
    )
    updated = _update_relator(x, active_n_gen, relator, max_length)
    return lax.cond(
        valid,
        lambda _: (updated, active_n_gen + 1),
        lambda _: (x, active_n_gen),
        operand=None,
    )


def _delete_generator(params, x: jnp.ndarray, active_n_gen: jnp.ndarray,
                      args: jnp.ndarray):
    """AC5: remove a generator with an isolated trivial relator."""
    max_length = params.max_length
    target = args[0]
    target_code = (target + 1).astype(x.dtype)

    def scan_slots(idx, carry):
        flags, contains_count = carry
        relator = _slice_relator(x, idx, max_length)
        rel_len = jnp.count_nonzero(relator)
        is_active = idx < active_n_gen
        is_trivial = is_active & (rel_len == 1) & (jnp.abs(relator[0]) == target_code)
        contains = is_active & jnp.any(jnp.abs(relator) == target_code)
        flags = flags.at[idx].set(is_trivial)
        contains_count = contains_count + contains.astype(jnp.int32)
        return flags, contains_count

    init_flags = jnp.zeros((params.max_n_gen,), dtype=jnp.bool_)
    trivial_flags, contains_count = lax.fori_loop(
        0, params.max_n_gen, scan_slots, (init_flags, jnp.array(0, dtype=jnp.int32))
    )
    relator_idx = jnp.argmax(trivial_flags)
    has_trivial_relator = jnp.any(trivial_flags)
    valid = (target < active_n_gen) & has_trivial_relator & (contains_count == 1)

    renumbered = _renumber_after_delete(x, target_code)

    def compact_body(idx, out):
        copy_slot = idx < (active_n_gen - 1)

        def copy(_):
            source_idx = jnp.where(idx < relator_idx, idx, idx + 1)
            relator = _slice_relator(renumbered, source_idx, max_length)
            return _update_relator(out, idx, relator, max_length)

        def clear(_):
            empty = jnp.zeros((max_length,), dtype=x.dtype)
            return _update_relator(out, idx, empty, max_length)

        return lax.cond(copy_slot, copy, clear, operand=None)

    compacted = lax.fori_loop(0, params.max_n_gen, compact_body, renumbered)
    return lax.cond(
        valid,
        lambda _: (compacted, active_n_gen - 1),
        lambda _: (x, active_n_gen),
        operand=None,
    )


# Cyclic normalization helpers shared by substitution and change-of-variables.
def rotate_relator_k(i: int, k: int, params, x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotates the i-th relator in x left by k positions (wraps around nonzero part).
    k can be any integer (rotation wraps around).
    """
    max_length = params.max_length
    start = i * max_length
    relator = lax.dynamic_slice(x, (start,), (max_length,))
    mask = relator != 0
    length = jnp.sum(mask).astype(jnp.int32)

    def rotate_nonzero(relator, mask, k):
        k_mod = k % length
        idx = (jnp.arange(max_length) + k_mod) % length
        idx = jnp.where(mask, idx, jnp.arange(max_length))
        rotated_relator = jnp.take(relator, idx)
        return rotated_relator

    rotated = lax.cond(
        length == 0,
        lambda _: relator,
        lambda _: rotate_nonzero(relator, mask, k),
        operand=None,
    )

    out = lax.dynamic_update_slice(x, rotated, (start,))
    return out


def booth_lex_min_rotation_masked(s):
    """
    JAX-compatible Booth's algorithm that only considers non-zero prefix of s.
    Assumes that padding (zeros) is at the end.
    Returns the index of the lex smallest rotation of the non-zero prefix.
    """

    L_full = s.shape[0]
    length = jnp.sum(s != 0)
    s2 = jnp.concatenate([s, s])  # doubled string

    f = -jnp.ones(2 * L_full, dtype=jnp.int32)
    k = 0

    def body(i, val):
        f, k = val
        j = f[i - k - 1]

        def cond_fun(loop_val):
            j, k = loop_val
            ijk = k + j + 1
            # Out-of-bound comparison is always "unequal"
            valid = (i < length) & (ijk < length)
            neq = jnp.logical_or(s2[i] != s2[ijk], ~valid)
            return (j != -1) & neq

        def body_fun(loop_val):
            j, k = loop_val
            ijk = k + j + 1
            k_new = jax.lax.select(
                (ijk >= length) | (s2[i] < s2[ijk]),
                i - j - 1,
                k
            )
            j = f[j]
            return j, k_new

        j, k = jax.lax.while_loop(cond_fun, body_fun, (j, k))

        def set_f(f, i, k, j):
            ijk = k + j + 1
            neq = (s2[i] != s2[k]) | (i >= length) | (k >= length)
            f_new = jax.lax.cond(
                (j == -1) & neq,
                lambda: f.at[i - k].set(-1),
                lambda: f.at[i - k].set(j + 1),
            )
            return f_new

        f = set_f(f, i, k, j)

        def new_k_fn():
            return jax.lax.select(
                (s2[i] < s2[k]) | (k >= length),
                i,
                k
            )

        k = jax.lax.cond(
            (j == -1) & ((s2[i] != s2[k]) | (i >= length) | (k >= length)),
            new_k_fn,
            lambda: k
        )

        return f, k

    f, k = jax.lax.fori_loop(1, 2 * L_full, body, (f, k))
    return k


# Substitution moves. `s_move` preserves the original two-relator behavior;
# `s_move_pair` is the generic n-generator version.
def s_move(i: int, params, x: jnp.ndarray, active_n_gen: jnp.ndarray,
           rk1k2: jnp.ndarray):
    """
    S-move:
    - Optionally inverts the second relator if r == 1 (for the computation only)
    - Rotates relator 1 by k1 and relator 2 by k2 (always left)
    - Multiplies (concatenates) the two relators
    - Substitutes the result into the i-th relator in x, leaving the other unchanged
    """
    max_length = params.max_length
    r, k1, k2 = rk1k2[0], rk1k2[1], rk1k2[2] # type: ignore

    # Step 1: Optionally invert the second relator for computation only
    def maybe_invert_for_comp(x):
        return _invert(1, params, x)
    x_comp = lax.cond(r == 1, maybe_invert_for_comp, lambda x: x, x)

    # Step 2: Rotate relator 1 and relator 2 (from possibly-inverted copy)
    x1_rot = rotate_relator_k(0, k1, params, x_comp)
    x2_rot = rotate_relator_k(1, k2, params, x_comp)

    # Step 3: Extract rotated relators
    rel1 = lax.dynamic_slice(x1_rot, (0,), (max_length,))
    rel2 = lax.dynamic_slice(x2_rot, (max_length,), (max_length,))

    # Step 4: Concatenate (multiply) relators
    x_concat = _concatenate(0, 1, params, jnp.concatenate([rel1, rel2]))

    # Step 5: Substitute result into i-th relator in the ORIGINAL x
    new_relator = lax.dynamic_slice(x_concat, (0,), (max_length,))
    equal = jnp.all(new_relator == rel1)
    booth_index = booth_lex_min_rotation_masked(new_relator)

    valid = active_n_gen == 2

    def skip_update(_):
        return x  # return original x unchanged

    def do_update(_):
        updated = lax.cond(
            i == 0,
            lambda _: lax.dynamic_update_slice(x, new_relator, (0,)),
            lambda _: lax.dynamic_update_slice(x, new_relator, (params.max_length,)),
            operand=None
        )
        return rotate_relator_k(i, booth_index, params, updated)

    out = lax.cond(valid & (~equal), do_update, skip_update, operand=None)
    return out, active_n_gen


def s_move_pair(target: int, source: int, params, x: jnp.ndarray,
                active_n_gen: jnp.ndarray, rk1k2: jnp.ndarray):
    """Generic substitution move for max_n_gen > 2."""
    max_length = params.max_length
    r, k1, k2 = rk1k2[0], rk1k2[1], rk1k2[2]  # type: ignore

    def maybe_invert_for_comp(x):
        return _invert(source, params, x)

    x_comp = lax.cond(r == 1, maybe_invert_for_comp, lambda x: x, x)
    x_target_rot = rotate_relator_k(target, k1, params, x_comp)
    x_source_rot = rotate_relator_k(source, k2, params, x_comp)
    rel_target = lax.dynamic_slice(x_target_rot, (target * max_length,), (max_length,))
    rel_source = lax.dynamic_slice(x_source_rot, (source * max_length,), (max_length,))
    x_concat = _concatenate(0, 1, params, jnp.concatenate([rel_target, rel_source]))
    new_relator = lax.dynamic_slice(x_concat, (0,), (max_length,))
    equal = jnp.all(new_relator == rel_target)
    booth_index = booth_lex_min_rotation_masked(new_relator)
    valid = (target < active_n_gen) & (source < active_n_gen) & (target != source) & (~equal)

    def do_update(_):
        updated = lax.dynamic_update_slice(x, new_relator, (target * max_length,))
        return rotate_relator_k(target, booth_index, params, updated)

    out = lax.cond(valid, do_update, lambda _: x, operand=None)
    return out, active_n_gen


def _branch_to_pair(branch: int, max_n_gen: int):
    target = branch // (max_n_gen - 1)
    source = branch % (max_n_gen - 1)
    if source >= target:
        source += 1
    return target, source


# Action registration: build the branch list used by `lax.switch` in ACS.
def setup_s_actions(params: EnvParams):
    """Package substitution branches."""
    jit_s_move = jax.jit(s_move, static_argnames=("i", "params"))
    if params.max_n_gen == 2:
        s_moves = [partial(jit_s_move, i, params) for i in range(params.max_n_gen)]
        return s_moves

    jit_s_move_pair = jax.jit(
        s_move_pair, static_argnames=("target", "source", "params")
    )
    s_moves = []
    for branch in range(num_substitution_branches(params.max_n_gen)):
        target, source = _branch_to_pair(branch, params.max_n_gen)
        s_moves.append(partial(jit_s_move_pair, target, source, params))
    return s_moves


def setup_ac_actions(params: EnvParams, change_of_variables_moves: bool = False,
                     ac45_moves: bool = False, stable_ac_moves: bool = False):
    """Package substitution actions plus optional stable AC branches."""
    if stable_ac_moves:
        change_of_variables_moves = True
        ac45_moves = True
    actions = setup_s_actions(params)
    if change_of_variables_moves:
        jit_cov_move = jax.jit(
            _change_of_variables,
            static_argnames=("remove_gen", "iso_relator", "params"),
        )
        cov_moves = [
            partial(jit_cov_move, remove_gen, iso_relator, params)
            for remove_gen in range(params.max_n_gen)
            for iso_relator in range(params.max_n_gen)
        ]
        actions = actions + cov_moves
    if ac45_moves:
        jit_add_generator = jax.jit(_add_generator, static_argnames=("params",))
        jit_delete_generator = jax.jit(_delete_generator, static_argnames=("params",))
        actions = (
            actions
            + [
                partial(jit_add_generator, params),
                partial(jit_delete_generator, params),
            ]
        )
    return actions
