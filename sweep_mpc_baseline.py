#!/usr/bin/env python3
"""Score Upkie's own MPC balancer on the full evaluation set.

This is the hand-engineered baseline our trained policies are compared against.
`upkie.controllers.MPCBalancer` ships inside the upkie package.

It is measured by the SAME code the trained policies are measured by --
`trainer_pybullet._rollout_episode` -- so the two numbers are comparable. The
only new thing here is an adapter that lets the MPC controller stand in for a
trained policy, because it needs the spine observation and SB3 policies do not.

    ./.venv/bin/python sweep_mpc_baseline.py             # a window, all 20
    ./.venv/bin/python sweep_mpc_baseline.py --headless   # numbers only
    ./.venv/bin/python sweep_mpc_baseline.py --speed 0.9  # one speed only

A fresh environment per episode, as RULES.md R4 requires, and a fresh
MPCBalancer with it -- the controller holds internal state and must not carry it
across episode boundaries any more than the simulator should.
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
    p.add_argument("--headless", action="store_true",
                   help="no window, numbers only. A window is the default.")
    p.add_argument("--wait", action="store_true",
                   help="open the window and wait for Enter before the first "
                        "episode. Start your screen recorder during the wait.")
    p.add_argument("--speed", type=float, default=None,
                   help="only run episodes at this commanded speed")
    p.add_argument("--out", default="out/mpc_baseline")
    args = p.parse_args()

    episodes = [e for e in eval_set.episodes()
                if args.speed is None or e["commanded_speed"] == args.speed]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gui = not args.headless
    print(f"Upkie MPC balancer, {len(episodes)} episodes "
          f"({'GUI window' if gui else 'headless'}), scored by "
          f"{scorer.name()} @ {scorer.scorer_sha()}")
    print(f"  starting tilt +-{reward_env.INIT_PITCH_RAD} rad, "
          f"push {eval_set.PUSH_NEWTONS} N\n")
    if gui:
        print("A PyBullet window should open. Ctrl-C to stop.\n")

    if args.wait and gui:
        # No window yet -- the first episode opens it. So: recorder first,
        # Enter, then the window appears and the robot starts immediately.
        print("\n  Start your screen recorder now, then press Enter here.")
        print("  The window opens as soon as you do.\n")
        try:
            input()
        except EOFError:
            pass

    records = []
    solver_failures = 0
    for index, episode in enumerate(episodes, 1):
        # R4: a fresh environment per episode, and a fresh controller with it.
        env = reward_env.make_env(_no_reward, gui=gui)
        wrapper = reward_env.find_reward_wrapper(env)
        tap = _InfoTap(env)
        policy = _MPCPolicy(tap, wrapper, dt=1.0 / reward_env.FREQUENCY)
        try:
            record = trainer_pybullet._rollout_episode(
                policy, tap, wrapper, episode)
        finally:
            solver_failures += policy.solver_failures
            env.close()
        records.append(record)
        print(f"  {index:2d}/{len(episodes)}  {episode['kind']:4s} "
              f"seed {episode['seed']:3d}  {episode['commanded_speed']} m/s  "
              f"{'FELL' if record['fell'] else 'up'} "
              f"{record['episode_seconds']:5.2f}s  "
              f"pitch {record['max_pitch_excursion_deg']:5.2f}deg  "
              f"disp {record['net_displacement_m']:+.2f}m")
        sys.stdout.flush()

    rollout = {"schema_version": trainer_pybullet.SCHEMA_VERSION
               if hasattr(trainer_pybullet, "SCHEMA_VERSION") else "2.0",
               "episodes": records}
    (out / "rollout.json").write_text(json.dumps(rollout, indent=2))
    result = scorer.score(rollout, "upkie_mpc_controller")
    # RULES.md R3: a result file records the code that made it.
    import subprocess  # noqa: PLC0415

    def _git(*a):
        return subprocess.run(["git", *a], capture_output=True,
                              text=True).stdout.strip()

    result["provenance"] = {
        "code_sha": _git("rev-parse", "HEAD"),
        "code_sha_short": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--short")),
        "init_pitch_rad": reward_env.INIT_PITCH_RAD,
        "push_newtons": eval_set.PUSH_NEWTONS,
        "episodes": len(records),
        "source": "upkie.controllers.MPCBalancer",
    }
    (out / "score.json").write_text(json.dumps(result, indent=2))

    print(f"\n  score {result['score']}")
    for key, value in result["measurements"].items():
        print(f"    {key:28} {value}")
    print(f"    {'violations':28} {result['violations'] or 'none'}")
    if solver_failures:
        print(f"\n  QP solver failed on {solver_failures} steps")
    print(f"\n  wrote {out}/rollout.json and {out}/score.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
