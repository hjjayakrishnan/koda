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


def starting_from_a_known_reward():
    """search_13's bug: --from-reward without its score loses the parent.

    r3c1 was handed in at 0.1468, the running best was left unset, round 1
    scored 0.0245, and `beats_best(0.0245, None)` said yes because there was
    nothing to compare against. Round 2 was bred from 0.0245 -- the search
    walked away from the very thing it was told to improve. Two paid calls went
    into it before it was caught.

    The test that existed only checked `beats_best` in isolation, which was
    always correct. What was missing was a test of what gets HANDED to it.
    """
    print("Starting from a known reward keeps that reward as the parent")

    # The bug, written out: no starting score.
    check("WITH NO starting score, a far worse round 1 takes over",
          beats_best(0.0245, None) is True,
          "this is correct for beats_best, and is why the score must be passed")

    # The fix: the starting score is carried in.
    check("WITH the starting score, a worse round 1 is rejected",
          beats_best(0.0245, 0.1468) is False)
    check("a round that genuinely beats the starting point is taken",
          beats_best(0.1600, 0.1468) is True)

    # run() must refuse rather than guess.
    import inspect

    from koda import orchestrator
    src = inspect.getsource(orchestrator.run)
    check("run() refuses --from-reward without --from-score",
          "from_reward and from_score is None" in src)
    check("it does not default the score to zero",
          "from_score or 0" not in src and "from_score, 0" not in src)

    # Replay search_13 as it should have gone.
    best = 0.1468           # r3c1, handed in
    for score in (0.0245,):  # what round 1 actually produced
        if beats_best(score, best):
            best = score
    check("replaying search_13: the parent stays 0.1468, not 0.0245",
          best == 0.1468, f"parent would be {best}")


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

    starting_from_a_known_reward()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print(f"{14} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
