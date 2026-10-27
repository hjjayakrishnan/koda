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

STEP_LIMIT = 2000  # 20 s at 100 Hz
PUSH_SPEED = 0.6  # commanded speed during a push episode

# The pre-registered shove. Fixed here, frozen at prereg-v1.
PUSH_AT_STEP = 500  # 5 s in, once the policy has settled
PUSH_NEWTONS = 25.0
PUSH_DURATION_S = 0.1
PUSH_DIRECTION = "forward"

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
