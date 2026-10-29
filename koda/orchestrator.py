# SPDX-License-Identifier: Apache-2.0
"""The loop. This file drives everything and writes the run record.

One round:

  1. build the prompt, then strip the robot's identity out of it
  2. get 6 candidate reward functions
  3. Gate 1: parse and call each one. The broken ones die here, cheaply
  4. train each survivor
  5. grade each trained candidate from its physical measurements alone
  6. write the reflection for the next round

Then round 2 and round 3, each starting from the best reward so far, its
reflection, and the exact error text of everything that died.

The run record is written to disk after every round, so a run that dies halfway
still renders a report.
"""

import json
import time
from pathlib import Path

from koda import SCHEMA_VERSION, candidates as candidates_mod
from koda import eval_set, gate1, prompt as prompt_mod, reflect, sanitize

# The two swap points.
#   scorer:  the real grader, red-teamed in tests/test_scorer.py. Swapped in
#            2026-10-26. `scorer_stub.py` is kept only so an old run record can
#            still be read back; no new run uses it.
#   trainer: trainer_stub -> trainer_pybullet, once real training is proven.
from koda import scorer  # noqa: F401
# The trainer is selected per run, not imported once. `trainer_stub` fakes the
# numbers so the loop can be exercised offline in seconds; `trainer_pybullet`
# is real PPO and takes about 17 minutes a candidate. Importing the stub at
# module level was how a "real" search could have called Nemotron for actual
# money and then fake-trained its output.
from koda import trainer_stub


def _pick_trainer(name: str):
    if name == "pybullet":
        from koda import trainer_pybullet  # noqa: PLC0415

        return trainer_pybullet
    if name == "stub":
        return trainer_stub
    raise ValueError(f"unknown trainer: {name!r}")


trainer = trainer_stub  # replaced per-run by run(); see _pick_trainer


def _dead_result(candidate_id, round_number, verdict):
    """A TrainerResult for a candidate that never reached the trainer."""
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "round": round_number,
        "trainer": trainer.name(),
        "status": verdict["status"],
        "error": verdict["error"],
        "terms": None,
        "rollout": None,
        "policy_path": None,
        "video_path": None,
        "train_seconds": 0.0,
        "compute_seconds": 0.0,
    }


def _pathologies(round_number, records):
    """Pathology classes 1 to 3, straight from the data.

    Class 4 -- a candidate that scores well and is visibly wrong -- is tagged by
    hand from the evaluation video. There is no video in Stage A, so class 4 is
    empty here rather than guessed at.
    """
    out = []
    for record in records:
        status = record["trainer"]["status"]
        if status in ("dead_compile", "dead_gate1"):
            out.append(
                {
                    "class": 1,
                    "round": round_number,
                    "candidate_id": record["candidate_id"],
                    "label": "did not survive Gate 1",
                    "detail": (record["trainer"]["error"] or "").strip().splitlines()[-1],
                    "passed_scorer": False,
                }
            )
        elif status in ("dead_nan", "dead_gate2"):
            out.append(
                {
                    "class": 2,
                    "round": round_number,
                    "candidate_id": record["candidate_id"],
                    "label": "reward or training went non-finite",
                    "detail": (record["trainer"]["error"] or "").strip().splitlines()[-1],
                    "passed_scorer": False,
                }
            )
        elif record["score"] and record["score"]["violations"]:
            out.append(
                {
                    "class": 3,
                    "round": round_number,
                    "candidate_id": record["candidate_id"],
                    "label": "scored "
                    + f"{record['score']['score']:.3f} and broke a hardware limit",
                    "detail": ", ".join(record["score"]["violations"]),
                    "passed_scorer": False,
                }
            )
    return out


def _grade_worker(payload):
    """Train and grade one candidate, in its own process.

    Has to be a module-level function taking only picklable arguments, so the
    trainer arrives as a NAME and is resolved here rather than passed as a
    module object.

    Two things are set per worker:

    * `torch.set_num_threads(1)`. Torch defaults to 8 on this machine, so six
      workers would put 48 threads on 16 cores and every run would be slower
      than running them one at a time.
    * Nothing else -- each process gets its own PyBullet physics server, which
      is the reason this is processes and not threads. `reward_env.make_env`
      warns that a second environment in the same process can disconnect the
      first one's server; separate processes sidestep that entirely rather than
      working around it.
    """
    (source, candidate_id, round_number, goal, seed, trainer_name, steps,
     run_dir) = payload
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
    except Exception:
        pass
    return _grade_one(
        source, candidate_id, round_number, goal, seed,
        _pick_trainer(trainer_name), steps, run_dir,
    )


def _grade_batch(batch, round_number, goal, seed, trainer_name, jobs, steps, run_dir):
    """Grade a round's candidates, `jobs` at a time, in candidate order.

    Candidates within a round are independent -- each trains a fresh policy
    from the same starting point -- so they parallelise cleanly. Rounds do not:
    round 2's prompt is built from round 1's winner.

    Results come back in the order they were submitted regardless of which
    finished first, so the winner on a tie does not depend on scheduling.
    """
    payloads = [
        (c["source"], c["candidate_id"], round_number, goal, seed, trainer_name,
         steps, run_dir)
        for c in batch
    ]
    if jobs <= 1 or len(payloads) == 1:
        return [_grade_worker(p) for p in payloads]

    import concurrent.futures as futures  # noqa: PLC0415

    with futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_grade_worker, payloads))


def _grade_one(source, candidate_id, round_number, goal, seed, trainer=trainer_stub,
               steps=None, run_dir="runs"):
    """Gate 1, then train, then grade. Returns one candidate record."""
    verdict = gate1.check(source, candidate_id)
    if not verdict["ok"]:
        return {
            "candidate_id": candidate_id,
            "trainer": _dead_result(candidate_id, round_number, verdict),
            "score": None,
        }

    kwargs = {"out_dir": run_dir}
    if steps is not None:
        kwargs["total_steps"] = steps
    result = trainer.train(
        verdict["reward_fn"], candidate_id, round_number, goal, seed, **kwargs
    )
    score = (
        scorer.score(result["rollout"], candidate_id)
        if result["status"] == "trained"
        else None
    )
    return {"candidate_id": candidate_id, "trainer": result, "score": score}


def _human_baseline(path, goal, seed, trainer=trainer_stub, run_dir="runs"):
    """Run the entrant's own hand-written reward through the identical path."""
    source = Path(path).read_text()
    record = _grade_one(source, "human_baseline", 0, goal, seed, trainer,
                        run_dir=run_dir)
    return {
        "source_path": str(path),
        "status": record["trainer"]["status"],
        "score": record["score"]["score"] if record["score"] else None,
        "measurements": record["score"]["measurements"] if record["score"] else None,
        "self_authored": True,
        "note": (
            "The entrant's own hand-written reward, written and frozen before any "
            "model output was seen. Disclosed as self-authored."
        ),
    }


def run(
    goal: str,
    robot: str = "upkie",
    rounds: int = 3,
    candidates: int = 6,
    out_dir: str = "out",
    seed: int = 0,
    search_id: str = "A_healthy_upkie",
    human_baseline_path: str | None = None,
    source: str = "nemotron",
    trainer_name: str = "pybullet",
    jobs: int = 1,
    steps: int | None = None,
) -> dict:
    """Run the whole search. Returns the run record and writes it to disk."""
    trainer = _pick_trainer(trainer_name)
    # Candidate ids repeat across searches -- seed 11 and seed 23 both produce
    # an "r1c3" -- so every search gets its own policy directory. Without this
    # the second search silently overwrites the first one's policies.
    run_dir = f"runs/seed{seed}"
    out = Path(out_dir)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    run_path = out / "run.json"
    started = time.time()

    record = {
        "schema_version": SCHEMA_VERSION,
        "search_id": search_id,
        "robot": robot,
        "env": "UpkieGyropod",
        "goal": goal,
        "config": {
            "candidates": candidates,
            "rounds": rounds,
            "seed": seed,
            "candidate_source": source,
            "jobs": jobs,
            "steps_per_candidate": steps,
            "policy_dir": run_dir,
            "trainer": trainer.name(),
            "scorer": scorer.name(),
            "candidate_source": "fixtures",
            "episodes_per_candidate": eval_set.EPISODE_COUNT,
        },
        "human_baseline": None,
        "rounds": [],
        "pathologies": [],
        "bill": {
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "compute_seconds": 0.0,
            "usd_compute": 0.0,
            "usd_tokens": 0.0,
            "note": "stub trainer and fixture candidates: nothing was spent",
        },
    }

    if human_baseline_path and Path(human_baseline_path).exists():
        record["human_baseline"] = _human_baseline(
            human_baseline_path, goal, seed, trainer, run_dir
        )  # baseline uses the trainer's own default step count

    best_source = None
    best_reflection = None
    previous_errors = []

    for round_number in range(1, rounds + 1):
        text = prompt_mod.build(
            goal=goal,
            candidates=candidates,
            best_reward_source=best_source,
            reflection=best_reflection,
            errors=previous_errors,
        )
        clean = sanitize.sanitize_or_raise(text)
        prompt_path = out / "prompts" / f"round{round_number}.txt"
        prompt_path.write_text(clean)

        if source == "nemotron":
            raw_path = out / "prompts" / f"round{round_number}_reply.txt"
            batch, call_meta = candidates_mod.from_nemotron(
                clean, round_number, candidates, save_raw=raw_path
            )
            # Every candidate's source is written to disk too, so a search is
            # reproducible from its own record without replaying the API.
            for candidate in batch:
                src = out / "candidates" / f"round{round_number}"
                src.mkdir(parents=True, exist_ok=True)
                (src / f"{candidate['candidate_id']}.py").write_text(
                    candidate["source"] + "\n"
                )
                candidate["source_path"] = str(
                    (src / f"{candidate['candidate_id']}.py").relative_to(out)
                )
            record["bill"]["prompt_tokens"] += call_meta["prompt_tokens"] or 0
            record["bill"]["completion_tokens"] += call_meta["completion_tokens"] or 0
            record["bill"]["model_calls"] += 1
            record["model"] = call_meta["model"]
        elif source == "fixtures":
            batch = candidates_mod.from_fixtures(round_number, candidates)
        else:
            raise ValueError(f"unknown candidate source: {source!r}")
        records = _grade_batch(
            batch, round_number, goal, seed, trainer_name, jobs, steps, run_dir
        )
        for graded, candidate in zip(records, batch):
            graded["source_path"] = candidate["source_path"]
            graded["source"] = candidate["source"]

        scored = [r for r in records if r["score"] is not None]
        winner = max(scored, key=lambda r: r["score"]["score"]) if scored else None

        round_record = {
            "round": round_number,
            "prompt_sha": prompt_mod.sha(clean),
            "prompt_path": str(prompt_path),
            "candidates": [
                {
                    "candidate_id": r["candidate_id"],
                    "source_path": r["source_path"],
                    "trainer": r["trainer"],
                    "score": r["score"],
                }
                for r in records
            ],
            "winner": winner["candidate_id"] if winner else None,
            "reflection": (
                reflect.reflect(winner["trainer"], winner["score"]) if winner else None
            ),
            "survivors": len(scored),
            "died": len(records) - len(scored),
        }
        record["rounds"].append(round_record)
        record["pathologies"] += _pathologies(round_number, records)
        record["bill"]["compute_seconds"] += sum(
            r["trainer"]["compute_seconds"] for r in records
        )
        run_path.write_text(json.dumps(record, indent=2))

        if winner:
            best_source = winner["source"]
            best_reflection = round_record["reflection"]
        previous_errors = [
            (r["candidate_id"], r["trainer"]["status"], r["trainer"]["error"])
            for r in records
            if r["trainer"]["error"]
        ]

    record["bill"]["compute_seconds"] = round(record["bill"]["compute_seconds"], 1)
    record["wall_seconds"] = round(time.time() - started, 2)
    run_path.write_text(json.dumps(record, indent=2))
    return record


def search_all(
    goal: str,
    seeds=(11,),
    robot: str = "upkie",
    rounds: int = 3,
    candidates: int = 6,
    out_dir: str = "out",
    source: str = "nemotron",
    trainer_name: str = "pybullet",
    jobs: int = 1,
    steps: int | None = None,
    human_baseline_path: str | None = None,
) -> dict:
    r"""Run the whole search once per seed, and compare the results.

    One search is `candidates` reward functions x `rounds` rounds. This runs X
    of those, where X is `len(seeds)`, each from a different random seed.

        seeds=(11,)            X=1   18 training runs,  3 model calls
        seeds=(11, 23, 37)     X=3   54 training runs,  9 model calls

    Why repeat it at all: the result moves with the seed. Two runs identical
    except for the seed differed by 12 seconds of total survival (measured
    2026-10-27). One search tells you what happened once, not whether it would
    happen again.

    Each search writes its own self-contained record to
    `out/search_<seed>/run.json` -- prompts, raw replies, every candidate's
    source, every traceback, every score, the winner, the reflection, the
    pathologies and the bill. Nothing has to be written down by hand.

    This function adds `out/summary.json` on top: the winning score from each
    search and the spread between them, which is the only thing that needs more
    than one search to say.
    """
    searches = []
    for seed in seeds:
        record = run(
            goal=goal,
            robot=robot,
            rounds=rounds,
            candidates=candidates,
            out_dir=str(Path(out_dir) / f"search_{seed}"),
            seed=seed,
            search_id=f"seed_{seed}",
            human_baseline_path=human_baseline_path,
            source=source,
            trainer_name=trainer_name,
            jobs=jobs,
            steps=steps,
        )
        searches.append(record)

    winners = []
    for record in searches:
        scored = [
            r["winner"] for r in record["rounds"] if r.get("winner")
        ]
        final = record["rounds"][-1].get("winner") if record["rounds"] else None
        winners.append(
            {
                "seed": record["config"]["seed"],
                "final_winner": final,
                "rounds_with_a_winner": len(scored),
            }
        )

    scores = []
    for record in searches:
        last = record["rounds"][-1] if record["rounds"] else None
        best = None
        if last:
            graded = [
                c["score"]["score"]
                for c in last["candidates"]
                if c.get("score")
            ]
            best = max(graded) if graded else None
        scores.append(best)

    present = [s for s in scores if s is not None]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "goal": goal,
        "robot": robot,
        "X": len(seeds),
        "seeds": list(seeds),
        "candidates_per_round": candidates,
        "rounds": rounds,
        "training_runs": len(seeds) * rounds * candidates,
        "candidate_source": source,
        "trainer": trainer_name,
        "jobs": jobs,
        "steps_per_candidate": steps,
        "winners": winners,
        "final_round_best_score": scores,
        "spread": (
            {
                "min": min(present),
                "max": max(present),
                "range": round(max(present) - min(present), 4),
            }
            if len(present) > 1
            else None
        ),
        "bill": {
            "model_calls": sum(r["bill"]["model_calls"] for r in searches),
            "prompt_tokens": sum(r["bill"]["prompt_tokens"] for r in searches),
            "completion_tokens": sum(
                r["bill"]["completion_tokens"] for r in searches
            ),
            "compute_seconds": round(
                sum(r["bill"]["compute_seconds"] for r in searches), 1
            ),
        },
        "note": (
            "X searches is a reproducibility check, not a significance test. "
            "Three samples give a range, not a confidence interval."
            if 1 < len(seeds) < 5
            else None
        ),
    }
    path = Path(out_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
