# search_15 — everything working at once, and it still did not beat r3c1

Run 2026-10-30, 02:01 to 03:12, from a clean tree at `6582f54`.

    ./.venv/bin/python -m koda.cli run --run-id search_15 \
        --from-reward results/search_12/candidates/round3/r3c1.py \
        --from-score 0.1981 \
        --source nemotron --trainer pybullet --steps 500000 --jobs 6 --seeds 11

3 paid calls, 5.43 compute-hours, 71 minutes wall clock.

This is the first search where nothing downstream was broken: a learner that
sees 5 s ahead, a grader that catches thrashing and standing still, seeds that
vary the start, one environment per scoring episode, and feedback numbers
describing real behaviour. Every earlier search had at least one of those wrong.

## Result

```
best per round:  0.0179  ->  0.0358  ->  0.0388
bar (r3c1, same settings, 500k):  0.1981
```

**Monotonic for the first time** — every round beat the one before, which no
previous search managed. And every round was still rejected as parent, because
none came close to the 0.1981 it started from.

| round | winner | score | survival | headroom | distance |
|---|---|---|---|---|---|
| 1 | r1c1 | 0.0179 | 0.359 | 0.953 | +0.008 |
| 2 | r2c4 | 0.0358 | 0.604 | 0.987 | +0.011 |
| 3 | r3c1 | 0.0388 | 0.267 | 0.438 | +0.413 |

## The finding

**Shown a good reward function and asked to improve it, the model writes a new
one rather than refining it — and new ones are far worse.**

Four searches have now handed `r3c1` to the model:

```
search_13 round 1   0.0245
search_14 round 1   0.0566
search_15 round 1   0.0179      <- working learner, honest feedback
```

Fixing the learner did not change this. It is not an artefact of the old
short-horizon agent; it survived the fix.

What the loop *can* do is climb from where it is: 0.0179 -> 0.0358 -> 0.0388
inside this run. What it does not do is take a 0.1981 function and make it
better. It restarts from its own ideas, which sit around 0.02-0.04.

`r3c1` itself was a round-3 reward from search_12 — produced by climbing, over
three rounds, from a cold start. That is consistent: the loop builds, it does
not polish.

## What this implies about scale

Eureka runs 5 restarts x 5 iterations x 16 samples = 400 candidates and averages
5 PPO runs per reward. If the mechanism here is "the loop climbs slowly from
wherever it starts", then more iterations is exactly the missing ingredient --
not a better prompt, and not a better learner, both of which are now fixed.

Three rounds climbed from 0.0179 to 0.0388, roughly doubling. r3c1 took three
rounds to reach 0.1468 under a broken learner. Neither is evidence of what
fifteen rounds would do, and nobody here has run fifteen rounds.

## Round 3 contains the familiar failure, once more

`r3c4`: survival **1.000**, headroom **0.995**, distance **-0.006**, score
**0.0000**. Perfect on both gates, travels six millimetres backwards. The
grader zeroes it. Under the additive grader this would have led the round.
