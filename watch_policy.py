#!/usr/bin/env python3
"""Watch a trained policy drive the robot.

Opens a PyBullet window by default so you can see what it learned.

    ./.venv/bin/python watch_policy.py                      # the latest policy
    ./.venv/bin/python watch_policy.py --speed 0.9
    ./.venv/bin/python watch_policy.py --policy runs/r1c1/policy.zip
    ./.venv/bin/python watch_policy.py --headless           # numbers only
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koda import candidates, gate1, reward_env  # noqa: E402


def newest_policy() -> Path | None:
    found = sorted(Path("runs").glob("*/policy.zip"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None,
                        help="path to a policy.zip (default: the newest in runs/)")
    parser.add_argument("--reward", default=None,
                        help="optional: the reward file this policy was trained "
                             "against. Only needed to SEE the reward value while "
                             "it runs -- the policy itself does not use it")
    parser.add_argument("--speed", type=float, default=0.3,
                        help="commanded ground speed in m/s (default 0.3)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--headless", action="store_true",
                        help="no window, just print the numbers")
    parser.add_argument("--no-follow", action="store_true",
                        help="do not move the camera with the robot")
    args = parser.parse_args()

    policy_path = Path(args.policy) if args.policy else newest_policy()
    if policy_path is None or not policy_path.exists():
        print("No policy found. Train one first:")
        print("    ./.venv/bin/python train_one.py")
        return 1

    # The policy maps observation -> action and never consults a reward. The
    # reward is only needed because RewardWrapper computes one every step, and
    # that wrapper is what appends commanded_speed to the observation. So when
    # no reward file is given, a no-op stands in and the reward reads zero.
    if args.reward:
        verdict = gate1.check(Path(args.reward).read_text(), "watched")
        if not verdict["ok"]:
            print(f"That reward file does not pass Gate 1: {verdict['error']}")
            return 1
        reward_fn = verdict["reward_fn"]
    else:
        reward_fn = lambda state, action: {}  # noqa: E731

    from stable_baselines3 import PPO  # noqa: PLC0415

    print(f"Policy:    {policy_path}")
    print(f"Reward:    {args.reward or '(none -- not needed to watch)'}")
    print(f"Commanded: {args.speed} m/s for {args.seconds:.0f} s")
    print(f"Mode:      {'headless' if args.headless else 'GUI window'}")
    if not args.headless:
        print("\nA PyBullet window should open. Ctrl-C to stop.\n")

    env = reward_env.make_env(
        reward_fn, gui=not args.headless, fixed_speed=args.speed
    )
    model = PPO.load(policy_path, device="cpu")

    steps = int(args.seconds * reward_env.FREQUENCY)
    observation, info = env.reset(seed=args.seed)

    # The robot drives away from a fixed camera -- at 0.9 m/s it leaves the
    # frame in a few seconds -- so keep the camera on it unless asked not to.
    # backend._bullet is a physics CLIENT ID (an int), not a module -- calling
    # methods on it raises AttributeError. The camera functions live on the
    # pybullet module and take the id as physicsClientId.
    client_id = None
    if not args.headless and not args.no_follow:
        try:
            import pybullet  # noqa: PLC0415

            client_id = int(
                reward_env.find_reward_wrapper(env).env.unwrapped.backend._bullet
            )
        except Exception:
            client_id = None  # still watchable, just a static camera
    worst_pitch = 0.0
    total_reward = 0.0
    fell = False
    done = 0

    try:
        for step in range(steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if client_id is not None and step % 10 == 0:
                forward = float(info["spine_observation"]["wheel_odometry"]["position"])
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=1.6,
                    cameraYaw=50,
                    cameraPitch=-20,
                    cameraTargetPosition=[forward, 0.0, 0.4],
                    physicsClientId=client_id,
                )
            worst_pitch = max(worst_pitch, abs(float(observation[0])))
            done = step + 1
            if terminated or truncated:
                fell = True
                break
    except KeyboardInterrupt:
        print("\nStopped.")

    seconds = done / reward_env.FREQUENCY
    spine = info["spine_observation"]
    print()
    print(f"  stayed up        {seconds:.2f} s of {args.seconds:.0f} s"
          f"{'  (FELL)' if fell else ''}")
    print(f"  worst pitch      {math.degrees(worst_pitch):.2f} deg")
    print(f"  ground covered   {float(spine['wheel_odometry']['position']):+.3f} m")
    print(f"  final speed      {float(spine['wheel_odometry']['velocity']):+.3f} m/s"
          f"  (asked for {args.speed})")
    print(f"  total reward     {total_reward:.1f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
