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

| id | score | stayed_upright | torque_headroom | speed error | mean moved |
|---|---|---|---|---|---|
| r2c3 | **0.0943** | 0.420 | 0.861 | 0.5064 | +2.382 m |
| r3c5 | 0.0870 | 0.618 | 0.921 | 0.5806 | +0.630 m |
| r1c1 | 0.0816 | 0.611 | 0.966 | 0.6265 | +0.465 m |
| r1c5 | 0.0440 | 0.371 | 0.537 | 0.7141 | +4.126 m |
| r1c3 | 0.0276 | 0.351 | 0.961 | 0.5894 | +0.246 m |
| r3c3 | 0.0270 | 0.241 | 0.503 | 0.7791 | +5.438 m |
| r2c1 | 0.0212 | 0.878 | 0.381 | 0.9715 | +2.684 m |
| r3c4 | 0.0072 | 0.320 | 0.142 | 0.9160 | +12.166 m |
| r1c2 | 0.0060 | 0.179 | 0.922 | 0.5983 | +0.035 m |
| r3c2 | 0.0026 | 1.000 | 0.041 | 1.0823 | +6.003 m |
| r2c4 | 0.0010 | 0.075 | 0.141 | 0.6156 | +0.313 m |
| r3c1 | 0.0008 | 0.077 | 0.294 | 0.6011 | +0.028 m |
| r2c6 | 0.0001 | 0.877 | 0.029 | 1.4661 | +3.272 m |
| r1c4 | 0.0000 | 0.262 | 0.947 | 0.5993 | +0.014 m |
| r1c6 | 0.0000 | 0.947 | 0.988 | 0.6279 | -0.511 m |
| r2c2 | 0.0000 | 0.307 | 0.058 | 0.8466 | -4.727 m |
| r2c5 | 0.0000 | 0.162 | 0.918 | 0.6141 | -0.255 m |

**Regraded 2026-10-29 at commit `447f8c7`**, scorer `333b1fa`, from the saved
rollouts with no re-simulation — `scorer.score()` is a pure function of a
rollout, so the physics is untouched. The grader now multiplies in
`torque_headroom` and takes a weighted geometric mean of speed accuracy and
distance covered, so staying upright by thrashing, and standing still, both stop
paying. What the change did:

| id | before | after | why |
|---|---|---|---|
| r3c2 | 0.3101 | 0.0026 | pinned its wheels at the torque limit on 96% of steps |
| r1c6 | 0.1076 | 0.0000 | drifted backward 0.51 m — covered no ground at all |
| r3c4 | 0.1502 | 0.0072 | drove 12 m at wildly wrong speeds |
| r2c3 | 0.1158 | 0.0943 | best candidate: travels 2.38 m, headroom 0.866 |
| MPC | 0.6031 | 0.5341 | barely touched — headroom 0.910 |

## What it says

**r3c2 never falls — and that was hiding how it does it.** Survival 1.000 across
all 20 episodes while travelling 6 m forward, achieved by sitting on the wheel
torque limit for 96% of every step. Watched in the GUI it visibly convulses. Its
`torque_headroom` is 0.041, so it now scores 0.0026 and ranks twelfth.

**r2c3 is the best candidate at 0.0943.** It travels 2.38 m with headroom 0.866.
It only survives 42% of the episode, so it is not good — it is merely the least
bad of the seventeen once thrashing and standing still stop paying.

Watch them:

    ./.venv/bin/python watch_policy.py --policy runs/seed11/r2c3/policy.zip
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
