# The learner was the bottleneck, not the reward functions

    ./.venv/bin/python zoo_hparams.py \
        --reward results/search_12/candidates/round3/r3c1.py --id r3c1_zoohp

Same reward function, same seed, same 1,000,000 steps as the control. The only
change is the PPO settings, taken from `rl_zoo3_balancer` — a third-party
SB3/PPO balancer for this same robot.

| | score | survival | torque_headroom | speed error | full 20 s episodes |
|---|---|---|---|---|---|
| r3c1, our settings, 500k | 0.1468 | 0.638 | 0.914 | 0.508 | — |
| r3c1, our settings, 1M | 0.1617 | 0.622 | 0.881 | 0.508 | — |
| **r3c1, rl_zoo3 settings, 1M** | **0.2286** | **0.920** | 0.898 | 0.532 | **12/20** |
| MPC, hand-built | 0.5341 | 1.000 | 0.928 | 0.283 | 20/20 |

Survival 0.622 -> 0.920 with the reward function untouched. This is the best
policy this project has produced, and no reward search produced it.

## What changed

```
                   rl_zoo3              ours (SB3 defaults)
n_timesteps        1_000_000            500_000
learning_rate      lin_0.003            0.0003
gamma              0.999                0.99
clip_range         0.1                  0.2
net_arch           pi[16,16] vf[32,32]  [64,64] both
normalize          enabled              not used
```

## Why gamma is the suspect

The effective horizon is about `1/(1-gamma)` control steps, at 200 Hz:

```
gamma 0.99   ->  ~100 steps  = 0.5 s     <- ours
gamma 0.999  ->  ~1000 steps = 5.0 s     <- theirs
```

This robot must drive its wheels BACKWARD for roughly half a second before it
can accelerate forward — it is non-minimum phase. At gamma 0.99 that manoeuvre
never pays off inside the horizon the agent can see, so the policy learns to
stay upright and not drive. "Balances but will not track speed" is the exact
behaviour every search here produced, and it is what a half-second-horizon agent
does on this plant.

## What it invalidates

Every reward function in this repo — 54 across three searches — was trained and
scored by this learner. `r3c1` scored highest not necessarily because it was the
best reward written, but because it was least harmed by a short-sighted agent.
The rankings in `results/search_11`, `search_12` and `search_14` measure the
reward and the learner together, and cannot separate them.

It also explains the noise. A learner that barely works gives erratic results,
which is how a 0.0560 parent produced a 0.1468 child in search_12 while a 0.1468
parent produced nothing above 0.1158 in search_14.

## Caveats, stated plainly

- One run, one seed. The same single-sample problem that made the searches noisy
  applies to this number too.
- Six settings changed at once. gamma is the hypothesis, normalisation second.
  Neither is isolated and the attribution is untested.
- It still falls in 8 of 20 episodes and tracks speed 1.9x worse than MPC.

`zoo_hparams.py` is deliberately standalone and nothing in `koda/` was changed.
Whether these settings become the project's defaults is a decision, not a
side effect of an experiment.
