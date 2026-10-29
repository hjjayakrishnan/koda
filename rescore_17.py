#!/usr/bin/env python3
"""Re-score the 17 saved search_11 policies on the fixed harness.

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python <this file>                 # all 17
    ./.venv/bin/python <this file> --only r3c2     # one, for a smoke test

No training. No Nebius. No money. It loads each saved policy, runs the 20
evaluation episodes with a FRESH environment per episode, and grades it with
the real scorer.

Records code_sha, dirty and a sha256 of every reward function, as RULES.md R3
requires -- so this file can be checked against the code that made it, which is
exactly what results/search_11/run.json cannot do.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from stable_baselines3 import PPO                      # noqa: E402
from koda import eval_set, gate1, reward_env, scorer    # noqa: E402
from koda import trainer_pybullet as T                  # noqa: E402

CANDIDATES = Path("results/search_11/candidates")
POLICIES = Path("runs/seed11")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True,
                          text=True).stdout.strip()


def find_source(candidate_id):
    for round_dir in sorted(CANDIDATES.iterdir()):
        path = round_dir / f"{candidate_id}.py"
        if path.exists():
            return path
    raise FileNotFoundError(f"no source for {candidate_id}")


def rescore(candidate_id, episodes_wanted):
    source = find_source(candidate_id)
    text = source.read_text()
    reward_fn = gate1.check(text, candidate_id)["reward_fn"]
    model = PPO.load(str(POLICIES / candidate_id / "policy.zip"), device="cpu")

    episodes = []
    for episode in episodes_wanted:
        env = reward_env.make_env(reward_fn, seed=episode["seed"], gui=False,
                                  max_steps=T.EVAL_STEPS)
        wrapper = reward_env.find_reward_wrapper(env)
        try:
            episodes.append(T._rollout_episode(model, env, wrapper, episode))
        finally:
            env.close()

    rollout = {"schema_version": T.SCHEMA_VERSION, "episodes": episodes}
    verdict = scorer.score(rollout, candidate_id)
    return {
        "candidate_id": candidate_id,
        "source_path": str(source),
        "source_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
        "score": verdict,
        "rollout": rollout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one candidate id")
    parser.add_argument("--out", default="out/rescore_17/rescore.json")
    args = parser.parse_args()

    ids = sorted(p.parent.name for p in POLICIES.glob("*/policy.zip"))
    if args.only:
        ids = [i for i in ids if i == args.only]
    episodes_wanted = eval_set.episodes()

    dirty = bool(git("status", "--short"))
    header = {
        "generated": "re-score of search_11 policies on the fixed harness",
        "code_sha": git("rev-parse", "HEAD"),
        "code_sha_short": git("rev-parse", "--short", "HEAD"),
        "dirty": dirty,
        "scorer": scorer.name(),
        "scorer_sha": scorer.scorer_sha(),
        "init_pitch_rad": reward_env.INIT_PITCH_RAD,
        "push_newtons": eval_set.PUSH_NEWTONS,
        "episodes_per_candidate": len(episodes_wanted),
        "candidates": len(ids),
    }
    print(json.dumps(header, indent=2))
    if dirty:
        print("\n*** WORKING TREE IS DIRTY -- R2: these numbers are "
              "UNCOMMITTED, not evidence ***\n")
    print(f"{len(ids)} policies x {len(episodes_wanted)} episodes\n")
    print(f"{'id':6s} {'score':>7s} {'survive':>8s} {'spd err':>8s} "
          f"{'moved':>8s} {'secs':>6s}  violations")
    print("-" * 72)

    started = time.time()
    results = []
    for n, candidate_id in enumerate(ids, 1):
        t0 = time.time()
        try:
            row = rescore(candidate_id, episodes_wanted)
        except Exception as error:                      # noqa: BLE001
            print(f"{candidate_id:6s}  FAILED  {type(error).__name__}: {error}")
            results.append({"candidate_id": candidate_id,
                            "error": f"{type(error).__name__}: {error}"})
            continue
        results.append(row)
        s = row["score"]
        m = s.get("measurements", {})
        task = [e for e in row["rollout"]["episodes"] if e["kind"] == "task"]
        moved = sum(e["net_displacement_m"] for e in task) / max(1, len(task))
        print(f"{candidate_id:6s} {s['score']:7.4f} "
              f"{m.get('survival_frac', 0):8.4f} "
              f"{m.get('mean_speed_error', 0):8.4f} "
              f"{moved:+8.3f} {time.time()-t0:6.0f}  "
              f"{','.join(s.get('violations') or []) or '-'}")
        sys.stdout.flush()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"header": header, "results": results}, out.open("w"),
                  indent=2)

    ok = [r for r in results if "score" in r]
    ok.sort(key=lambda r: r["score"]["score"], reverse=True)
    print("-" * 72)
    print(f"done in {(time.time()-started)/60:.1f} min   "
          f"{len(ok)} scored, {len(results)-len(ok)} failed")
    if ok:
        print("\nranked:")
        for r in ok:
            print(f"   {r['candidate_id']:6s} {r['score']['score']:.4f}")
        print(f"\nbest: {ok[0]['candidate_id']} at "
              f"{ok[0]['score']['score']:.4f}")
        print("MPC human baseline on the OLD harness was 0.5348 -- not "
              "comparable, the harness changed.")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
