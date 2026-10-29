#!/usr/bin/env python3
"""The seed must change where the robot starts.

    ./.venv/bin/python tests/test_seeding.py

Required by RULES.md R5. It exists because for the whole of the first search
`env.reset(seed=...)` ignored the seed: every episode began at exactly
[0, 0, 0, 0, setpoint]. The 20-episode evaluation set is "5 seeds x 3 speeds",
so it was really 3 distinct episodes repeated five times, and no number this
project produced carried any variance information. Nothing failed, nothing
warned, and two people reasoned on top of it for two days.

So it is checked mechanically from now on.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koda import eval_set, reward_env  # noqa: E402

FAILURES = []


def check(label, passed, detail=""):
    print(f"  {'ok  ' if passed else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not passed:
        FAILURES.append(label)


def start_state(seed):
    """The observation the policy sees on its first step, for one seed."""
    env = reward_env.make_env(lambda state, action: {}, fixed_speed=0.3,
                              max_steps=10)
    observation, _info = env.reset(seed=seed)
    env.close()
    return np.asarray(observation, dtype=np.float64)


def main():
    print("The seed changes where the robot starts")

    starts = {seed: start_state(seed) for seed in eval_set.SEEDS}

    distinct = {tuple(np.round(v, 9)) for v in starts.values()}
    check(f"all {len(eval_set.SEEDS)} evaluation seeds give different starts",
          len(distinct) == len(eval_set.SEEDS),
          f"{len(distinct)} distinct")

    for seed, state in starts.items():
        check(f"seed {seed} does not start perfectly upright",
              abs(state[0]) > 1e-6, f"pitch={state[0]:+.6f} rad")

    first = eval_set.SEEDS[0]
    again = start_state(first)
    check(f"seed {first} is exactly reproducible",
          np.array_equal(starts[first], again),
          f"{starts[first][0]:+.9f} vs {again[0]:+.9f}")

    worst = max(abs(v[0]) for v in starts.values())
    check("every start is within the approved tilt",
          worst <= reward_env.INIT_PITCH_RAD + 1e-9,
          f"max |pitch|={worst:.5f} <= {reward_env.INIT_PITCH_RAD}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print(f"{len(eval_set.SEEDS) + 3} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
