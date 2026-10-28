#!/usr/bin/env python3
"""Score Upkie's own MPC balancer on the full evaluation set.

This is the hand-engineered baseline our trained policies are compared against.
`upkie.controllers.MPCBalancer` ships inside the upkie package.

It is measured by the SAME code the trained policies are measured by --
`trainer_pybullet._rollout_episode` -- so the two numbers are comparable. The
only new thing here is an adapter that lets the MPC controller stand in for a
trained policy, because it needs the spine observation and SB3 policies do not.

    ./.venv/bin/python sweep_mpc_baseline.py            # all 20 episodes
    ./.venv/bin/python sweep_mpc_baseline.py --gui       # watch it
    ./.venv/bin/python sweep_mpc_baseline.py --speed 0.9 # one speed only
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from koda import eval_set, reward_env, scorer, trainer_pybullet
from upkie.controllers import MPCBalancer


def _no_reward(state, action):
    """MPC needs no reward. RewardWrapper requires a callable, so: none."""
    return {}


class _InfoTap(gym.Wrapper):
    """Keeps the latest `info` so the MPC adapter can read the spine."""

    def __init__(self, env):
        super().__init__(env)
        self.last_info = {}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_info = info
        return observation, info

    def step(self, action):
        out = self.env.step(action)
        self.last_info = out[4]
        return out


class _MPCPolicy:
    """Makes MPCBalancer look like an SB3 policy: .predict(obs) -> action."""

    def __init__(self, tap, wrapper, dt):
        self.tap = tap
        self.wrapper = wrapper
        self.dt = dt
        self.balancer = MPCBalancer()
        self.solver_failures = 0

    def predict(self, observation, deterministic=True):
        spine = self.tap.last_info.get("spine_observation")
        if spine is None:  # before the first step of an episode
            return np.array([0.0], dtype=np.float32), None
        try:
            velocity = self.balancer.step(
                target_ground_velocity=float(self.wrapper.commanded_speed),
                spine_observation=spine,
                dt=self.dt,
            )
        except Exception:  # a QP that does not solve is a real failure mode
            self.solver_failures += 1
            velocity = 0.0
        return np.array([velocity], dtype=np.float32), None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gui", action="store_true", help="open a window and watch")
    p.add_argument("--speed", type=float, default=None,
                   help="only run episodes at this commanded speed")
    p.add_argument("--out", default="out/mpc_baseline")
    args = p.parse_args()

    episodes = [e for e in eval_set.episodes()
                if args.speed is None or e["commanded_speed"] == args.speed]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Upkie MPC balancer, {len(episodes)} episodes "
          f"({'GUI' if args.gui else 'headless'}), scored by "
          f"{scorer.name()} @ {scorer.scorer_sha()}\n")
    if args.gui:
        print("A PyBullet window should open. Ctrl-C to stop.\n")

    env = reward_env.make_env(_no_reward, gui=args.gui)
    wrapper = reward_env.find_reward_wrapper(env)
    tap = _InfoTap(env)
    policy = _MPCPolicy(tap, wrapper, dt=1.0 / reward_env.FREQUENCY)

    records = []
    try:
        for index, episode in enumerate(episodes, 1):
            record = trainer_pybullet._rollout_episode(policy, tap, wrapper, episode)
            records.append(record)
            print(f"  {index:2d}/{len(episodes)}  {episode['kind']:4s} "
                  f"seed {episode['seed']:3d}  {episode['commanded_speed']} m/s  "
                  f"{'FELL' if record['fell'] else 'up'} "
                  f"{record['episode_seconds']:5.2f}s  "
                  f"pitch {record['max_pitch_excursion_deg']:5.2f}deg  "
                  f"disp {record['net_displacement_m']:+.2f}m")
    finally:
        env.close()

    rollout = {"schema_version": trainer_pybullet.SCHEMA_VERSION
               if hasattr(trainer_pybullet, "SCHEMA_VERSION") else "2.0",
               "episodes": records}
    (out / "rollout.json").write_text(json.dumps(rollout, indent=2))
    result = scorer.score(rollout, "upkie_mpc_controller")
    (out / "score.json").write_text(json.dumps(result, indent=2))

    print(f"\n  score {result['score']}")
    for key, value in result["measurements"].items():
        print(f"    {key:28} {value}")
    print(f"    {'violations':28} {result['violations'] or 'none'}")
    if policy.solver_failures:
        print(f"\n  QP solver failed on {policy.solver_failures} steps")
    print(f"\n  wrote {out}/rollout.json and {out}/score.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
