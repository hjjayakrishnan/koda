# Koda — Eureka on a mini PC

## Inspiration

NVIDIA's [Eureka](https://arxiv.org/abs/2310.12931) showed a large language model can write
reward functions that beat human experts on 83% of 29 robotics tasks. It did that with
IsaacGym on GPUs, 400 reward candidates per robot, and five training runs averaged per
candidate.

We wanted to know what survives when you take that method to a laptop. No GPU, 16 CPU
threads, PyBullet, and roughly 1/22nd of the search budget. Not a demo of the method
working — a measurement of what it costs to find out.

## What it does

```
koda run --run-id search_15 --goal "balance upright and drive forward at the commanded speed"
```

1. **Nemotron 3 Super** (on Nebius Token Factory) is shown Upkie's environment with its
   reward line removed, the goal in one sentence, and the function signature. It writes 6
   reward functions.
2. **PPO trains each one** in PyBullet on a real wheeled biped model — 500,000 steps,
   6 candidates in parallel, one torch thread each.
3. **A grader the model never sees** scores each trained policy on 20 fixed episodes:
   three commanded speeds, five random starts, and five shove tests.
4. **The measurements go back to the model**, which writes the next round from the best
   reward so far.

Three rounds, three model calls, 18 trained policies, about 80 minutes.

## The result

**The hand-written controller won, 2.3x.**

| | score | survives | motor headroom | speed error |
|---|---|---|---|---|
| **MPC, hand-written** (Upkie's own) | **0.5341** | 1.000 | 0.928 | 0.283 m/s |
| **Koda's best** | **0.2286** | 0.920 | 0.898 | 0.532 m/s |
| Best first-attempt reward, of 16 tried | 0.0380 | 1.000 | 0.774 | 0.644 m/s |

That is the headline and we have not buried it. What we found getting there is the part
worth reading.

## What we found

**1. The learner was the bottleneck, not the reward search.**

PPO's discount factor `gamma` sets how far ahead the agent optimises. At 200 Hz,
Stable-Baselines3's default of 0.99 is a **0.5 second** horizon. An Upkie is non-minimum
phase: to drive forward it must first drive its wheels *backward* for about half a second
to tip the body over. At 0.99 that manoeuvre never pays off inside the horizon the agent
can see, so it learns to stay upright and not drive.

Same reward function, same seed, only the PPO settings changing:

```
500k steps, SB3 defaults          0.1468    survives 0.638
1M   steps, SB3 defaults          0.1617    survives 0.622
500k steps, rl_zoo3 settings      0.1981    survives 0.765
1M   steps, rl_zoo3 settings      0.2286    survives 0.920
```

**+56%, with no change to any reward function.** Every reward the search had ranked was
ranked by that learner, so every ranking measured the reward and the learner together.

**2. The model builds, it does not polish.**

Four times we handed the model its own best reward and asked it to improve on it. It
scored 0.0245, 0.0566, 0.0179 — against a 0.1981 bar. It rewrites rather than refines, and
rewrites are far worse. Fixing the learner did not change this.

But *within* a run it climbs: 0.0179 → 0.0358 → 0.0388, monotonic. And our best reward is
itself a round-3 reward, reached by climbing from a cold start. Of 16 first-attempt rewards
retrained under the fixed learner, the best scores 0.0380 — five times worse.

So the iteration is where the value is, and **three rounds is not many. Eureka runs 400
candidates across 25 iterations. We ran 18 across 3.** That ratio, not the prompt and not
the model, is the difference we can point at.

**3. A grader that only counts falls will reward thrashing and standing still.**

Our first grader added survival, speed and distance together. Under it, the top candidate
sat **pinned at the wheel torque limit for 96% of every control step** — it balanced by
vibrating. The video shows it.

The grader now multiplies two gates in:

```
score = stayed_upright x torque_headroom
        x speed_accuracy^0.55 x distance_covered^0.45 x straightness
```

A weighted *geometric* mean rather than an average, so zero distance cannot be bought back
with anything else, and by AM-GM it can never score higher than the average it replaced.
Re-scored, that candidate fell from 0.3101 to 0.0026 and placed twelfth of seventeen.

## How we built it

Python, Stable-Baselines3 PPO, PyBullet, Upkie's own robot model and controllers.
Nemotron 3 Super via Nebius Token Factory with an OpenAI-compatible client. The prompt is
anonymised before it is sent — the robot's name and motor part numbers are stripped.

The grader is deliberately blind to motion smoothness in the reward-shaping sense: a list
of forbidden terms (`jerk`, `action_rate`, `energy`, `smoothness`) is enforced by a test, so
the model has to invent those itself rather than have them handed over. Torque saturation
is scored because it is a hardware limit that was already measured, not a shaping choice.

## Challenges

**We found nine defects in our own measurement code.** A dead random seed that made all
five seeds identical, so "20 episodes" was 3 episodes repeated. A 25 N force that leaked
between episodes when a robot fell at the wrong moment. A test suite that made real paid
API calls because a default changed under it. Result files that said "nothing was spent"
on runs that cost money.

All nine are listed in `DEFECTS.md` with the commit that closed each — including the two we
closed too early and had to reopen, which produced the rule we now work under: **fix the
code, not the file it produced.**

Our first full search is **retracted in its own file**: it does not reproduce under the
code that produced it and we never found out why.

## The bill

Published in full, failed runs included.

```
14 model calls    22,685 prompt tokens    49,187 completion tokens    ~34 CPU-hours
model     nvidia/nemotron-3-super-120b-a12b, Nebius Token Factory
machine   AMD Ryzen 7 5800H, 8 cores / 16 threads, 27 GB RAM, no GPU
```

Two of the 14 calls were wasted on our own bug: a flag set the starting reward without its
score, so round 1's winner replaced it whatever it scored. Caught mid-run, killed, fixed,
and covered by a regression test. One run's record under-reports its calls by one and we
say so in the README rather than quietly correcting it.

## What we are not claiming

- No searched reward beat the hand-written controller. 0.2286 against 0.5341.
- One seed, one training run per number. Eureka averages five. Everything here is
  single-sample and the round-to-round noise is visible in the results.
- Three rounds is not a trend. One search went 0.0740 → 0.0560 → 0.1468: a worse parent
  produced a better child, and a better parent later produced worse children.
- Nothing ran on hardware. No physical Upkie was involved.

## What's next

The one ingredient never tried is **more iterations**. Three rounds roughly doubled the
score. Eureka runs twenty-five. Everything else — the learner, the grader, the seeds, the
feedback — is now fixed and tested; the search budget is the remaining variable, and it is
the one a laptop cannot buy.

## Built with

`python` `stable-baselines3` `pybullet` `upkie` `nvidia-nemotron-3-super` `nebius-token-factory` `ppo`
