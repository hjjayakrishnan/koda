# SPDX-License-Identifier: Apache-2.0
"""The command line.

    koda run --robot upkie --goal "balance upright and drive forward at the commanded speed"

Writes:

    out/run.json              every candidate, every score, every error
    out/report.html           the same thing, readable
    out/prompts/roundN.txt    exactly what was sent, after sanitizing
"""

import argparse
import sys
from pathlib import Path

from koda import orchestrator, report

GOAL = "balance upright and drive forward at the commanded speed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="koda", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the reward search")
    run.add_argument(
        "--run-id",
        required=True,
        help="a name for THIS search, e.g. search_12. Policies go to "
             "runs/<run-id>/search_<seed>/seed<seed>/, the record to "
             "out/<run-id>/. Required, and the run refuses to start if that "
             "name is already used: candidate ids are r1c1..r3c6 in every "
             "search, so without it a second search overwrites the first.",
    )
    run.add_argument(
        "--from-reward",
        default=None,
        help="path to a reward .py to START from. Round 1 improves this one "
             "instead of starting from nothing, so a finished search's winner "
             "can be carried into the next search.",
    )
    run.add_argument("--robot", default="upkie", help="which robot (only upkie so far)")
    run.add_argument("--goal", default=GOAL, help="the goal, in one sentence")
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--candidates", type=int, default=6)
    run.add_argument(
        "--seeds",
        default="11",
        help="comma-separated seeds. One search per seed, so the count IS X. "
             "`--seeds 11` is X=1 (18 training runs); `--seeds 11,23,37` is "
             "X=3 (54 runs). Repeating matters because the result moves with "
             "the seed.",
    )
    run.add_argument(
        "--steps",
        type=int,
        default=None,
        help="PPO steps per candidate. Default is the trainer's own (30,000). "
             "500,000 is what produced a policy that learned anything, at "
             "about 17 min a candidate.",
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="how many candidates in a round train at once, each in its own "
             "process. Rounds stay sequential -- round 2 needs round 1's "
             "winner. Default 6, which is one per candidate.",
    )
    run.add_argument(
        "--trainer",
        choices=("pybullet", "stub"),
        default="pybullet",
        help="`pybullet` is real PPO, about 17 min a candidate. `stub` fakes "
             "the numbers in seconds, for exercising the loop offline.",
    )
    run.add_argument(
        "--source",
        choices=("nemotron", "fixtures"),
        default="nemotron",
        help="where candidate reward functions come from. `nemotron` calls "
             "Nebius Token Factory and costs credits; `fixtures` replays "
             "hand-written files offline for debugging.",
    )
    run.add_argument("--out", default="out", help="where to write the run")
    run.add_argument("--search-id", default="A_healthy_upkie")
    run.add_argument(
        "--human-baseline",
        default="human_baseline.py",
        help="the entrant's own hand-written reward, if it exists yet",
    )

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error("unknown command")

    if args.robot != "upkie":
        parser.error(f"only --robot upkie is supported, not {args.robot!r}")

    try:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    except ValueError:
        parser.error(f"--seeds must be integers, got {args.seeds!r}")
    if not seeds:
        parser.error("--seeds needs at least one seed")

    runs_planned = len(seeds) * args.rounds * args.candidates
    print(
        f"X={len(seeds)} search(es), seeds {list(seeds)}: "
        f"{args.candidates} candidates x {args.rounds} rounds x {len(seeds)} "
        f"= {runs_planned} training runs, "
        f"{len(seeds) * args.rounds} {args.source} call(s), trainer={args.trainer}, jobs={args.jobs}\n"
    )

    summary = orchestrator.search_all(
        goal=args.goal,
        run_id=args.run_id,
        from_reward=args.from_reward,
        seeds=seeds,
        robot=args.robot,
        rounds=args.rounds,
        candidates=args.candidates,
        out_dir=args.out,
        source=args.source,
        trainer_name=args.trainer,
        jobs=args.jobs,
        steps=args.steps,
        human_baseline_path=args.human_baseline,
    )

    # The per-search records are the data; this is the only cross-search fact.
    if summary["spread"]:
        s = summary["spread"]
        print(f"\nacross {summary['X']} searches: best score "
              f"{s['min']:.4f} to {s['max']:.4f}, range {s['range']:.4f}")
        if summary["note"]:
            print(f"  note: {summary['note']}")

    bill = summary["bill"]
    print(f"\nbill: {bill['model_calls']} model calls, "
          f"{bill['prompt_tokens']:,} prompt + {bill['completion_tokens']:,} "
          f"completion tokens, {bill['compute_seconds'] / 3600:.2f} compute-hours")
    print(f"wrote {Path(args.out) / 'summary.json'}")
    return 0


def _unused_single_run_reporting(args, record):
    """Kept only so the per-round printing below is not lost."""
    html_path = Path(args.out) / "report.html"
    report.write(record, html_path)

    print(f"trainer: {record['config']['trainer']}   scorer: {record['config']['scorer']}")
    for round_record in record["rounds"]:
        winner = round_record["winner"]
        score = None
        for candidate in round_record["candidates"]:
            if candidate["candidate_id"] == winner and candidate["score"]:
                score = candidate["score"]["score"]
        print(
            f"round {round_record['round']}: "
            f"{round_record['survivors']} of {len(round_record['candidates'])} survived, "
            f"winner {winner} at {score if score is None else format(score, '.3f')}"
        )
    print(f"pathologies logged: {len(record['pathologies'])}")
    print(f"run record:  {Path(args.out) / 'run.json'}")
    print(f"report:      {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
