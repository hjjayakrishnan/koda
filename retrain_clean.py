#!/usr/bin/env python3
"""Retrain the reward functions that were never shown a false measurement.

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python <this file>

Round 1 of every search saw no measurement block -- a cold start sees nothing,
and --from-reward passes the reward SOURCE without its reflection. Rounds 2 and
3 were written in response to numbers produced by the old short-horizon learner,
so they answer a question that was not real. These 18 are the clean set.

Trained through koda.trainer_pybullet, which now carries rl_zoo3_balancer's
settings, and scored by koda.scorer. No model calls.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.getcwd())

STEPS = 500_000
JOBS = 6
OUT = Path("runs/clean18")

SOURCES = []
for folder, tag in (
    ("results/search_12/candidates/round1", "s12"),
    ("results/search_14/candidates/round1", "s14"),
    ("out/search_13/seed11/candidates/round1", "s13"),
):
    for f in sorted(Path(folder).glob("*.py")):
        SOURCES.append((f"{tag}_{f.stem}", str(f)))


def one(payload):
    name, path = payload
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from koda import gate1, scorer
    from koda import trainer_pybullet as T

    started = time.time()
    verdict = gate1.check(Path(path).read_text(), name)
    if not verdict["ok"]:
        return {"name": name, "path": path, "status": "dead_gate1"}
    result = T.train(verdict["reward_fn"], name, 1, "balance upright and drive "
                     "forward at the commanded speed", seed=11,
                     total_steps=STEPS, out_dir=str(OUT))
    if result["status"] != "trained":
        return {"name": name, "path": path, "status": result["status"]}
    s = scorer.score(result["rollout"], name)
    return {"name": name, "path": path, "status": "trained",
            "score": s, "minutes": round((time.time()-started)/60, 1)}


def main():
    print(f"{len(SOURCES)} reward functions, {STEPS:,} steps each, "
          f"{JOBS} at a time")
    print("settings: gamma 0.999, lin_0.003, clip 0.1, normalise on\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        for r in pool.map(one, SOURCES):
            rows.append(r)
            if r["status"] == "trained":
                m = r["score"]["measurements"]
                print(f"  {r['name']:12s} {r['score']['score']:7.4f}  "
                      f"surv {m['survival_frac']:.3f}  "
                      f"headroom {1-m['saturated_step_frac_wheel']:.3f}  "
                      f"err {m['mean_speed_error']:.3f}  "
                      f"[{r['minutes']} min]")
            else:
                print(f"  {r['name']:12s} {r['status']}")
            sys.stdout.flush()
            json.dump(rows, (OUT/"results.json").open("w"), indent=2)
    ok = [r for r in rows if r["status"] == "trained"]
    ok.sort(key=lambda r: -r["score"]["score"])
    print(f"\ndone in {(time.time()-started)/60:.0f} min, "
          f"{len(ok)}/{len(rows)} trained")
    print("\nranked:")
    for r in ok:
        print(f"  {r['name']:12s} {r['score']['score']:.4f}   {r['path']}")
    print("\nr3c1 with these settings at 500k was 0.1981, at 1M 0.2286")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
