#!/usr/bin/env python3
"""Train one reward function from fixtures/ with real PPO in real PyBullet.

    ./.venv/bin/python train_one.py
    ./.venv/bin/python train_one.py --steps 50000 --reward fixtures/round1/c6_distance_only.py

CPU only. There is no NVIDIA hardware on this machine and torch is the CPU build.
Roughly 185 simulation steps per second, so --steps 150000 takes about 13 minutes
plus the evaluation episodes.

Writes runs/<id>/policy.zip and out/train_one/result.json, then tells you the
command to watch what it learned.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koda import gate1, trainer_pybullet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", default="fixtures/round1/c1_pitch_and_speed.py")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=None,
                        help="how many eval episodes (default: all 20)")
    parser.add_argument("--id", default=None, help="name for runs/<id>/")
    args = parser.parse_args()

    reward_path = Path(args.reward)
    candidate_id = args.id or reward_path.stem

    verdict = gate1.check(reward_path.read_text(), candidate_id)
    print(f"Reward:  {reward_path}")
    print(f"Gate 1:  {verdict['status']}")
    if not verdict["ok"]:
        print(verdict["error"])
        return 1
    print(f"Terms:   {', '.join(verdict['term_names'])}")
    print(f"Steps:   {args.steps:,} at about 185 steps/s "
          f"-> roughly {args.steps / 185 / 60:.0f} min of training")
    print()

    started = time.time()
    result = trainer_pybullet.train(
        verdict["reward_fn"], candidate_id, 1, "balance upright and drive "
        "forward at the commanded speed", seed=args.seed,
        total_steps=args.steps, eval_episodes=args.episodes,
    )

    out = Path("out/train_one")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))

    print(f"status: {result['status']}  in {time.time() - started:.0f} s")
    if result["error"]:
        print(result["error"][-800:])
        return 1

    terms = result["terms"]
    print("\nreward terms, averaged per step, at 10/25/50/100% of training:")
    width = max(len(k) for k in terms["values"])
    for key, values in sorted(terms["values"].items()):
        print(f"  {key:<{width}}  " + "  ".join(f"{v:+8.4f}" for v in values))
    print(f"  {'(sum)':<{width}}  " + "  ".join(f"{v:+8.4f}" for v in terms["total"]))

    episodes = result["rollout"]["episodes"]
    stood = sum(1 for e in episodes if not e["fell"])
    print(f"\nevaluation: {stood} of {len(episodes)} episodes ran the full 20 s")
    for e in episodes[:6]:
        print(f"  {e['commanded_speed']} m/s seed {e['seed']:3d}: "
              f"up {e['episode_seconds']:5.2f} s, "
              f"speed error {e['mean_speed_error']:.3f} m/s, "
              f"moved {e['net_displacement_m']:+.3f} m, "
              f"worst pitch {e['max_pitch_excursion_deg']:5.1f} deg")

    print(f"\npolicy: {result['policy_path']}")
    print("\nWATCH IT:")
    print(f"  ./.venv/bin/python watch_policy.py --policy {result['policy_path']} "
          f"--reward {reward_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
