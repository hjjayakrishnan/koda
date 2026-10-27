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

# The two swap points. Stage B changes these two lines and nothing else.
#   trainer: trainer_stub -> trainer_pybullet
#   scorer:  scorer_stub  -> scorer   (written and owned by the other pane)
from koda import scorer_stub as scorer
from koda import trainer_stub as trainer


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


def _grade_one(source, candidate_id, round_number, goal, seed):
    """Gate 1, then train, then grade. Returns one candidate record."""
    verdict = gate1.check(source, candidate_id)
    if not verdict["ok"]:
        return {
            "candidate_id": candidate_id,
            "trainer": _dead_result(candidate_id, round_number, verdict),
            "score": None,
        }

    result = trainer.train(verdict["reward_fn"], candidate_id, round_number, goal, seed)
    score = (
        scorer.score(result["rollout"], candidate_id)
        if result["status"] == "trained"
        else None
    )
    return {"candidate_id": candidate_id, "trainer": result, "score": score}


def _human_baseline(path, goal, seed):
    """Run the entrant's own hand-written reward through the identical path."""
    source = Path(path).read_text()
    record = _grade_one(source, "human_baseline", 0, goal, seed)
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
) -> dict:
    """Run the whole search. Returns the run record and writes it to disk."""
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
            "trainer": trainer.name(),
            "scorer": scorer.name(),
            "candidate_source": "fixtures",
            "episodes_per_candidate": eval_set.EPISODE_COUNT,
        },
        "human_baseline": None,
        "rounds": [],
        "pathologies": [],
        "bill": {
            "compute_seconds": 0.0,
            "usd_compute": 0.0,
            "usd_tokens": 0.0,
            "note": "stub trainer and fixture candidates: nothing was spent",
        },
    }

    if human_baseline_path and Path(human_baseline_path).exists():
        record["human_baseline"] = _human_baseline(human_baseline_path, goal, seed)

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

        batch = candidates_mod.from_fixtures(round_number, candidates)
        records = []
        for candidate in batch:
            graded = _grade_one(
                candidate["source"],
                candidate["candidate_id"],
                round_number,
                goal,
                seed,
            )
            graded["source_path"] = candidate["source_path"]
            graded["source"] = candidate["source"]
            records.append(graded)

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
