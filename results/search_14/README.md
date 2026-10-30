# search_14 — started from search_12's winner, and never beat it

Run 2026-10-29, 21:23 to 22:33, from a clean tree at commit `19f6846`.

    ./.venv/bin/python -m koda.cli run --run-id search_14 \
        --from-reward results/search_12/candidates/round3/r3c1.py \
        --from-score 0.1468 \
        --source nemotron --trainer pybullet --steps 500000 --jobs 6 --seeds 11

3 rounds x 6 candidates, 3 paid calls, 5.05 compute-hours, 69 minutes wall clock.

## The result

| round | winner | score | became_parent |
|---|---|---|---|
| 1 | r1c4 | 0.0566 | **False** |
| 2 | r2c3 | 0.1158 | **False** |
| 3 | r3c4 | 0.0664 | **False** |

Started from `r3c1` at **0.1468**. Nothing beat it. Every round was correctly
rejected as parent, so all three rounds were bred from `r3c1` itself and the
search never walked backwards — the guard added in `8260572` worked exactly as
intended on every round.

It also never moved forward.

## What it shows, read together with search_12

```
search_12   parent 0.0560  ->  child 0.1468
search_14   parent 0.1468  ->  best child 0.1158
```

A worse parent produced a better child; a better parent produced worse children.
That is not what an improving search looks like. The most economical reading is
that round-to-round differences are mostly **noise**, and 0.1468 was a high draw
rather than a peak the loop climbed to.

Across `search_13` and `search_14` the model was handed `r3c1` and asked to beat
it in four separate rounds. Best produced: 0.1158. It never came close.

## Why this is probably not the model's fault

Eureka, the method being reproduced, runs 5 restarts x 5 iterations x 16 samples
= **400 candidates per robot** and averages **5 independent PPO runs** per final
reward. We ran 18 candidates scored by a single run each — about 1/22nd of the
search, with single-sample scoring where they average five.

More importantly, `rl_zoo3_balancer` — a third-party SB3/PPO balancer for this
exact robot — uses `gamma: 0.999`, `learning_rate: lin_0.003`, observation
normalisation and the `Pendulum-Shaped` environment. We use `gamma: 0.99`, a
learning rate 10x smaller, no normalisation. At 200 Hz, gamma 0.99 gives an
effective horizon of about 0.5 s on a 20 s episode, on a plant that must drive
its wheels backwards for roughly that long before it can accelerate forwards.

So the "balances but won't track speed" behaviour these searches keep producing
may be a property of the learner, not of the reward functions being searched.
That is the next thing to test, and it is free.

## Note on the bill

`run.json` carries the corrected billing note — "3 PAID call(s) to
nvidia/nemotron-3-super-120b-a12b" — the first live run since the hardcoded
"nothing was spent" was fixed at the source in `02e71d4`.
