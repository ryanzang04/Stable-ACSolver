"""Demo for the stable change-of-variables AC move.

Run from the repository root:

    python scripts/stable_cov_demo.py

The default example is the move from the note:
AK(3), choose z = xyx in the first relator, isolate x from yxy,
then remove x. After the move, the old x slot is interpreted as z.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from envs.ac_moves import setup_ac_actions
from envs.ac_s import EnvParams
from envs.utils import convert_relators_to_presentation, encode_action


INPUT_LETTERS = {1: "x", -1: "X", 2: "y", -2: "Y"}
OUTPUT_REMOVE_X = {1: "z", -1: "Z", 2: "y", -2: "Y"}
OUTPUT_REMOVE_Y = {1: "x", -1: "X", 2: "z", -2: "Z"}


def word_to_string(word, alphabet):
    letters = [alphabet[int(v)] for v in word if int(v) != 0]
    return "".join(letters) or "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="AK(n) exponent")
    parser.add_argument("--max_length", type=int, default=24)
    parser.add_argument("--remove_gen", type=int, choices=[0, 1], default=0,
                        help="0 removes x and reuses x's slot as z; 1 removes y")
    parser.add_argument("--iso_relator", type=int, choices=[0, 1], default=0)
    parser.add_argument("--z_start", type=int, default=0)
    parser.add_argument("--z_len", type=int, default=3)
    parser.add_argument("--z_inverse", action="store_true",
                        help="use z = W^{-1} instead of z = W")
    args = parser.parse_args()

    L = args.max_length
    r1 = [1, 2, 1, -2, -1, -2]  # xyx = yxy
    r2 = [1] * args.n + [-2] * (args.n + 1)  # x^n = y^(n+1)
    x = jnp.asarray(convert_relators_to_presentation(r1, r2, L))

    params = EnvParams(n_gen=2, max_length=L, max_steps_in_episode=1)
    actions = setup_ac_actions(params, change_of_variables_moves=True)
    branch = 2 + args.remove_gen * 2 + args.iso_relator
    action = jnp.asarray(
        [branch, int(args.z_inverse), args.z_start, args.z_len - 1],
        dtype=jnp.int32,
    )
    y, _ = jax.lax.switch(
        action[0], actions, x, jnp.array(params.n_gen, dtype=jnp.int32),
        action[1:],
    )

    out_alphabet = OUTPUT_REMOVE_X if args.remove_gen == 0 else OUTPUT_REMOVE_Y
    print("input:")
    print("  r1 =", word_to_string(x[:L], INPUT_LETTERS))
    print("  r2 =", word_to_string(x[L:], INPUT_LETTERS))
    print("action:")
    print("  vector =", [int(v) for v in action])
    print("  packed =", encode_action(action, L, change_of_variables_moves=True))
    print("output:")
    print("  r1 =", word_to_string(y[:L], out_alphabet))
    print("  r2 =", word_to_string(y[L:], out_alphabet))


if __name__ == "__main__":
    main()
