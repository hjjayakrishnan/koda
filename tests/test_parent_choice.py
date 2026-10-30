#!/usr/bin/env python3
"""The model must be shown the best reward so far, not the last round's.

    ./.venv/bin/python tests/test_parent_choice.py

This exists because of search_12. Round 1 scored 0.0740, round 2 scored 0.0560,
and round 3 was bred from round 2 -- the worse parent -- because the loop simply
took whatever the previous round produced. Round 3 recovered to 0.1468 anyway,
which was luck, not design.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koda.orchestrator import beats_best  # noqa: E402

FAILURES = []


def check(label, passed, detail=""):
    print(f"  {'ok  ' if passed else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not passed:
        FAILURES.append(label)


def main():
    print("The parent is the best reward so far, not the last round's")

    check("the first scored round always becomes the parent",
          beats_best(0.0740, None) is True)
    check("a better round replaces it",
          beats_best(0.1468, 0.0740) is True)
    check("a WORSE round does not replace it",
          beats_best(0.0560, 0.0740) is False)
    check("an equal round does not replace it",
          beats_best(0.0740, 0.0740) is False)
    check("a round with no scored candidate does not replace it",
          beats_best(None, 0.0740) is False)
    check("a round with no scored candidate and no parent yet stays empty",
          beats_best(None, None) is False)

    # search_12's actual sequence, replayed.
    best = None
    parents = []
    for score in (0.0740, 0.0560, 0.1468):
        if beats_best(score, best):
            best = score
        parents.append(best)
    check("replaying search_12: round 3 would have been bred from 0.0740, "
          "not 0.0560", parents[1] == 0.0740, f"parents seen: {parents}")
    check("the best is never lost", best == 0.1468, str(best))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("8 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
