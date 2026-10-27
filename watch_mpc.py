#!/usr/bin/env python3
"""Watch Upkie's own MPC balancer drive at a commanded speed.

This is the hand-engineered controller we compare against. It ships inside the
upkie package as upkie.controllers.MPCBalancer.

Opens a PyBullet window by default so you can watch it.

    ./.venv/bin/python watch_mpc.py                 # 0.3 m/s, 20 s, window
    ./.venv/bin/python watch_mpc.py --speed 0.9
    ./.venv/bin/python watch_mpc.py --headless      # no window, numbers only
"""

import argparse
import sys

import gymnasium as gym
import numpy as np

import upkie.envs
from upkie.controllers import MPCBalancer

FREQUENCY = 200.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speed", type=float, default=0.3,
                   help="commanded ground speed in m/s (default 0.3)")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="how long to run (default 20)")
    p.add_argument("--seed", type=int, default=11, help="episode seed")
    p.add_argument("--headless", action="store_true",
                   help="no window, just print the numbers")
    args = p.parse_args()

    upkie.envs.register()
    steps = int(args.seconds * FREQUENCY)
    dt = 1.0 / FREQUENCY

    print(f"Upkie's MPC balancer, commanded {args.speed} m/s, "
          f"{args.seconds:.0f} s, {'headless' if args.headless else 'GUI window'}.")
    if not args.headless:
        print("A PyBullet window should open. Ctrl-C to stop.\n")

    balancer = MPCBalancer()
    worst_pitch = 0.0
    fell = False
    done = 0

    with gym.make("Upkie-PyBullet-Pendulum", frequency=FREQUENCY,
                  gui=not args.headless, frequency_checks=False) as env:
        observation, info = env.reset(seed=args.seed)
        for step in range(steps):
            velocity = balancer.step(
                target_ground_velocity=args.speed,
                spine_observation=info["spine_observation"],
                dt=dt,
            )
            observation, _r, terminated, truncated, info = env.step(
                np.array([velocity], dtype=np.float32)
            )
            done = step + 1
            pitch = abs(float(info["spine_observation"]["base_orientation"]["pitch"]))
            worst_pitch = max(worst_pitch, pitch)
            if terminated or truncated:
                fell = terminated
                break

    seconds = done / FREQUENCY
    print(f"  {'FELL after' if fell else 'stayed up'} {seconds:.2f} s "
          f"of {args.seconds:.0f} s")
    print(f"  worst pitch {np.degrees(worst_pitch):.2f} deg")
    print(f"  commanded {args.speed} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
