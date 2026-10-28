# SPDX-License-Identifier: Apache-2.0
"""The evaluation episodes. Frozen at `git tag prereg-v1`.

Both trainers run exactly this list, so a stub score and a real score are
comparable shapes. No exploration noise in any of them.

20 episodes per candidate:

  15 task episodes   5 fixed seeds x 3 commanded speeds
   5 push episodes   the same 5 seeds, one pre-registered shove each

Commanded yaw rate is zero everywhere, so any heading drift is the policy's
doing and not the command's.
"""

SEEDS = (11, 23, 37, 53, 71)
SPEEDS = (0.3, 0.6, 0.9)

# Episode length and push timing are given in SECONDS, not steps.
#
# They were in steps, at an assumed 100 Hz, and the trainer runs at 200 Hz -- so
# `PUSH_AT_STEP = 500` put the shove at 0.5 s instead of 5 s, and every push
# episode reported "recovered" because the robot had not been disturbed yet.
# Seconds are rate-independent and that whole class of bug goes away.

EPISODE_SECONDS = 20.0
PUSH_SPEED = 0.6  # commanded speed during a push episode

# The pre-registered shove. Fixed here, frozen at prereg-v1.
PUSH_AT_SECONDS = 5.0  # once the policy has settled
PUSH_NEWTONS = 25.0
PUSH_DURATION_S = 0.1
PUSH_DIRECTION = "forward"


def step_limit(frequency: float) -> int:
    """Episode length in control steps, at whatever rate the trainer runs."""
    return int(EPISODE_SECONDS * frequency)


def push_step(frequency: float) -> int:
    """Which control step the shove lands on, at a given rate."""
    return int(PUSH_AT_SECONDS * frequency)

# Recovery means pitch back inside this band and staying there for the window.
RECOVERY_PITCH_RAD = 0.15
RECOVERY_HOLD_S = 1.0


def episodes() -> tuple:
    """Every evaluation episode, as plain dicts. Order is fixed."""
    out = []
    for seed in SEEDS:
        for speed in SPEEDS:
            out.append(
                {
                    "kind": "task",
                    "seed": seed,
                    "commanded_speed": speed,
                    "commanded_yaw_rate": 0.0,
                }
            )
    for seed in SEEDS:
        out.append(
            {
                "kind": "push",
                "seed": seed,
                "commanded_speed": PUSH_SPEED,
                "commanded_yaw_rate": 0.0,
            }
        )
    return tuple(out)


EPISODE_COUNT = len(episodes())
