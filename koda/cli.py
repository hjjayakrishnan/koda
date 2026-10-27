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
    run.add_argument("--robot", default="upkie", help="which robot (only upkie so far)")
    run.add_argument("--goal", default=GOAL, help="the goal, in one sentence")
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--candidates", type=int, default=6)
    run.add_argument("--seed", type=int, default=0)
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

    record = orchestrator.run(
        goal=args.goal,
        robot=args.robot,
        rounds=args.rounds,
        candidates=args.candidates,
        out_dir=args.out,
        seed=args.seed,
        search_id=args.search_id,
        human_baseline_path=args.human_baseline,
    )

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
