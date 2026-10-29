# Re-score of the 17 search_11 policies on the fixed harness

Produced 2026-10-29 by `rescore_17.py` at commit `7b96671`, clean tree.

    cd /home/jayakrishnan/Robotics/koda_v2
    ./.venv/bin/python rescore_17.py

No training, no Nebius, no money. It loads each saved policy from
`runs/seed11/<id>/policy.zip`, runs the 20 evaluation episodes with a fresh
environment per episode, and grades it with `koda/scorer.py`.

`rescore.json` carries `code_sha`, `dirty`, `scorer_sha`, `init_pitch_rad`,
`push_newtons` and a `source_sha` per candidate, as RULES.md R3 requires. That
is the difference between this file and `results/search_11/run.json`, which is
retracted and cannot be checked against the code that produced it.

## Result

| id | score | was | survival | was | speed err | mean moved |
|---|---|---|---|---|---|---|
| r3c2 | **0.3101** | 0.1779 | **1.000** | 0.689 | 1.0823 | +6.003 m |
| r3c4 | 0.1502 | 0.1163 | 0.320 | 0.255 | 0.9160 | +12.166 m |
| r2c6 | 0.1231 | 0.1245 | 0.877 | 0.783 | 1.4661 | +3.272 m |
| r2c3 | 0.1158 | 0.1401 | 0.420 | 0.532 | 0.5064 | +2.382 m |
| r3c5 | 0.1104 | 0.1244 | 0.618 | 0.865 | 0.5806 | +0.630 m |
| r1c6 | 0.1076 | 0.1197 | 0.947 | 1.000 | 0.6279 | -0.511 m |
| r1c5 | 0.1000 | 0.0703 | 0.371 | 0.465 | 0.7141 | +4.126 m |
| r2c1 | 0.0996 | 0.1368 | 0.878 | 0.841 | 0.9715 | +2.684 m |
| r1c1 | 0.0939 | 0.0988 | 0.611 | 0.707 | 0.6265 | +0.465 m |
| r3c3 | 0.0829 | 0.0769 | 0.241 | 0.237 | 0.7791 | +5.438 m |
| r1c3 | 0.0513 | 0.0508 | 0.351 | 0.351 | 0.5894 | +0.246 m |
| r1c4 | 0.0342 | 0.0481 | 0.262 | 0.364 | 0.5993 | +0.014 m |
| r1c2 | 0.0238 | 0.0500 | 0.179 | 0.378 | 0.5983 | +0.035 m |
| r2c5 | 0.0197 | 0.0248 | 0.162 | 0.200 | 0.6141 | -0.255 m |
| r3c1 | 0.0102 | 0.0097 | 0.077 | 0.073 | 0.6011 | +0.028 m |
| r2c4 | 0.0100 | 0.0095 | 0.075 | 0.071 | 0.6156 | +0.313 m |
| r2c2 | 0.0096 | 0.0084 | 0.307 | 0.312 | 0.8466 | -4.727 m |

The "was" columns come from `results/search_11/run.json`. They are **retracted**
and are shown only to say how far the broken harness was off. They are not a
baseline and must not be quoted as one.

## What it says

**r3c2 is the only policy of the 17 that never falls.** Survival 1.000 across
all 20 episodes, from five different starting tilts, while travelling 6 m
forward. The old harness scored it 0.1779 and it looked unremarkable.

Watch it:

    ./.venv/bin/python watch_policy.py --policy runs/seed11/r3c2/policy.zip

**Best per round is still monotonic:** r1c6 0.1076, r2c6 0.1231, r3c2 0.3101.
The round-3 improvement is large rather than marginal.

**The round-2 ranking changed.** r2c3 was the winner at 0.1401 and is now second
at 0.1158, behind r2c6. r2c3 is the candidate that fed round 3, so the fixed
harness would have carried a different parent forward.

**"Surviving by standing still" is mostly gone.** On the old harness nine
candidates held the full 20 s while barely moving. With a 0.02 rad starting tilt
most of them fall: r1c2 went from 0.378 survival to 0.179. They were not
balancing, they were starting perfectly balanced.

## Not comparable to the MPC baseline

`results/mpc_baseline_score.json` says 0.5348, measured on the old harness with
no starting tilt and a 25 N shove. The harness has changed in both respects. The
MPC controller has to be re-scored under `rescore_17.py`'s conditions before any
candidate-versus-baseline claim is made.
