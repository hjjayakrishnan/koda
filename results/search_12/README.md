# search_12 — the first search graded by the fixed scorer

Run 2026-10-29, 15:33 to 16:50, from a clean tree at commit `f5d25d7`.

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python -m koda.cli run --run-id search_12 --source nemotron \
        --trainer pybullet --steps 500000 --jobs 6 --seeds 11

3 rounds x 6 candidates = 18 PPO runs at 500,000 steps each. 3 paid calls to
`nvidia/nemotron-3-super-120b-a12b`, one per round. 6.99 compute-hours, 76
minutes wall clock at `--jobs 6`.

Policies are in `runs/search_12/seed11/<candidate>/policy.zip`.

## What makes this one different

Every candidate before it was graded by a scorer that paid for standing still
and for thrashing. This is the first search where

- staying upright is gated by `torque_headroom`, so sitting on the wheel torque
  limit cannot buy a score
- speed and distance are combined as a weighted geometric mean, so a robot that
  covers no ground scores zero no matter how steady it is
- the seed actually randomises the starting tilt
- each scoring episode gets a fresh environment
- the model is told, from round 2 on, what share of steps it spent pinned at
  the torque limit

## Result

| | score | survival | torque_headroom | speed error | distance |
|---|---|---|---|---|---|
| **r3c1** | **0.1468** | 0.638 | 0.914 | 0.508 | +0.162 |
| r3c4 | 0.0819 | 0.846 | 0.906 | 0.616 | +0.044 |
| r1c6 (round 1 winner) | 0.0740 | 0.472 | 0.825 | 0.591 | +0.137 |
| r1c1 | 0.0620 | 0.360 | 0.964 | 0.592 | +0.128 |
| r2c1 (round 2 winner) | 0.0560 | 0.295 | 0.828 | 0.551 | +0.158 |

Best per round: **0.0740 -> 0.0560 -> 0.1468**. Round 2 regressed and round 3
recovered past both. Three rounds is far too few to call that a trend.

Watch the winner:

    ./.venv/bin/python watch_policy.py \
        --policy runs/search_12/seed11/r3c1/policy.zip \
        --reward results/search_12/candidates/round3/r3c1.py --speed 0.6

## Against everything else measured on the same grader

| | score | survival | torque_headroom | speed error |
|---|---|---|---|---|
| MPC, hand-built | **0.5341** | 1.000 | 0.910 | 0.2831 |
| r3c1, this search | 0.1468 | 0.638 | 0.914 | 0.5080 |
| r2c3, best of search_11 | 0.0943 | 0.420 | 0.866 | 0.5064 |
| r3c2 + setpoint ramp | 0.0310 | 0.630 | 0.293 | 0.8053 |
| r3c2, search_11's "winner" | 0.0026 | 1.000 | 0.041 | 1.0823 |

`r3c1` beats the whole previous search by 56%, and beats it on mechanism rather
than on luck: `torque_headroom` 0.914 against MPC's 0.910 means it is as smooth
as the hand-engineered controller. It loses on survival (0.638 vs 1.000) and on
speed tracking (0.508 vs 0.283).

The gap to the hand-built baseline narrowed from 5.7x to 3.6x. It is still a
large gap and this is still not a working controller.

## The grader caught the old failure mode live

Round 1 contained two candidates that never fall:

    r1c2   survival 1.000   distance -0.479   score 0.0000
    r1c4   survival 1.000   distance -0.511   score 0.0000

Both stand still and drift backwards. Under the additive scorer these would have
looked like the strongest candidates in the round — that is exactly how
`search_11`'s `r3c2` won. Here they score zero, and a candidate that survives
47% while actually moving won instead.

## Honest limits

- One seed, one search, three rounds. Nothing here is a trend.
- Round 2 was worse than round 1. The loop is not monotonic.
- The reward terms shifted towards actuation cost in round 2 — `actuation_cost`,
  `mechanical_power`, `metabolic_cost`, `rate_smooth` — after the saturation
  line first appeared in the prompt, and the round 1 winner it was shown
  contained no such term. That is suggestive and nothing more. Round 1 had no
  feedback at all and still produced `action_smooth`, `torque_penalty`,
  `effort` and `efficiency`, so the model clearly does not need telling that
  effort matters.
- The `reflection` strings in `run.json` carry the old violation name
  `never_recovered_from_push`, which was renamed afterwards. They are left as
  they are: that is the exact text the model was sent.
