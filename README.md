# Koda

**Give it a robot and one sentence. Get back the reward function and a trained robot.**

Koda is [Eureka](https://arxiv.org/abs/2310.12931) reproduced on a mini PC — **no GPU, 16
CPU threads, 1/22nd of Eureka's search budget.** NVIDIA Nemotron 3 Super writes reward
functions on Nebius Token Factory, PPO trains each one in PyBullet, a grader the model
never sees scores the result, and the numbers go back to the model.

**It did not beat the hand-written controller.** The useful part is what we found while
measuring why, and all of it is in this repo beside the code that produced it.

```
koda run --robot upkie --goal "balance upright and drive forward at the commanded speed"
```

```
out/
  reward_terms.py     the reward function, written by NVIDIA Nemotron 3 Super
  policy.pt           the trained policy
  report.html         every candidate, every score, the model's reasoning, the videos
  bill.txt            the exact cost, failed runs included
```

---

## Running it

The loop itself needs nothing installed — Python 3.10 or newer and the standard
library:

```
python3 -m koda.cli run --robot upkie --goal "balance upright and drive forward at the commanded speed"
```

After `pip install -e .` the same thing is spelled `koda run --robot upkie --goal "..."`.

That command writes `out/run.json`, `out/report.html` and the exact prompt text
sent for each round. By default it runs on the hand-written candidates in
`fixtures/` and the stub trainer, which means no API call, no simulator and no
money — and the report says so on its own first line. Training against PyBullet
needs the extras: `pip install -e '.[sim]'`.

```
python3 tests/test_loop.py      # 169 checks, a few seconds
```

---

## Why this problem exists

Upkie's simulation environments ship with no reward function. The maintainers wrote
them, then deliberately removed them and left the decision to whoever writes the
agent. From `upkie/envs/upkie_env.py`:

```python
reward = 0.0  # to be decided by agents, for instance via wrappers
```

That comment is the specification for this project. Koda fills the hole, using the
extension point the comment itself names: a reward wrapper, not a patch to the
environment.

**One sentence on how this differs from the nearest similar work:** Stress Twin breaks
policies that already exist. Koda writes the objective for a robot that has none.

---

## How it works

1. Build a prompt from the environment source — the reward hole — plus the goal sentence.
2. Strip the robot's identity from the prompt text before sending it (`sanitize.py`).
3. **Nemotron 3 Super**, on **Nebius Token Factory**, returns 6 candidate reward functions.
4. **Gate 1** parses and unit-tests each candidate against a mock environment. No
   simulator, no GPU. Syntax errors, NaN, wrong tensor shapes and missing return keys
   die here, before any compute is spent. Runs locally by default, or fans the 6 out
   across **Nebius Serverless Jobs**.
5. **Gate 2 and training** — each survivor gets a smoke test, then PPO from scratch in
   PyBullet on **Nebius AI Cloud**. A crash sends its traceback back to the model so it
   fixes its own bug.
6. A grader the model never sees scores each trained policy.
7. Per-term training logs become a written report for the next round. Three rounds.
8. The whole search runs three times with different seeds.

6 candidates x 3 rounds x 3 seeded repeats = 54 training runs, plus controls.

### The prompt is anonymised before it is sent

Upkie's reward functions are public, so the model may have memorised one. Telling a
model "do not copy the published reward" cannot be verified and raises the salience of
the thing being avoided, so Koda removes the identity from the input instead: the robot
name and the mjbots motor part numbers are substituted out of the prompt text. The
simulator, the URDF and the physics are untouched.

This reduces the chance of recall. It does not prove recall did not happen, so the
resemblance of the model's output to the published reward is measured structurally and
reported either way.

---

## Which NVIDIA models were used

| Model | Where it runs | What it does |
|---|---|---|
| **NVIDIA Nemotron 3 Super** | Nebius Token Factory | Writes every candidate reward function, and rewrites them across rounds from the grader's feedback |

The model is given the environment source and a goal sentence. It is never shown the
grader's code — only grade values. Every reward function in this repository that is not
`human_baseline.py` or a deliberate test fixture was written by Nemotron.

## How Nebius Token Factory and AI Cloud helped

| Component | Platform service |
|---|---|
| Nemotron 3 Super inference | Nebius Token Factory |
| Gate 1 candidate checking, 6 in parallel | Nebius Serverless Jobs |
| All policy training and scoring | Nebius AI Cloud, CPU instance |

**Correction, stated plainly:** an earlier draft of this file said training ran on Nebius
AI Cloud. It did not. Nebius Token Factory served every Nemotron call — that part is real
and the token counts are below — but **all PPO training ran locally on a 16-thread AMD
Ryzen 7 5800H with no GPU.** PyBullet is CPU-bound, so a rented GPU would not have helped;
what the hardware limits is how many reward candidates can be tried, which is the whole
subject of Finding 2.

---

## The robot

[Upkie](https://github.com/upkie/upkie) — an open-source, 3D-printed wheeled biped,
Apache-2.0. Six actuated joints:

| Joints | Actuator | URDF effort limit |
|---|---|---|
| hip, knee (x4) | mjbots qdd100 | 16.0 Nm |
| wheel (x2) | mjbots mj5208 | 1.7 Nm |

Both limits are read from `upkie_description/urdf/upkie.urdf`. The wheels are the
balance effector, and at 1.7 Nm they are the tightest constraint on the machine.

---

## Results

Every number below was produced by committed code and is stamped with the commit that made
it. All of it re-scores from this repo with no model calls.

### The scoreboard

| | score | survives | motor headroom | speed error | full 20 s episodes |
|---|---|---|---|---|---|
| **MPC, hand-written** (Upkie's own) | **0.5341** | 1.000 | 0.928 | 0.283 m/s | 20/20 |
| **Koda's best**, `r3c1` | **0.2286** | 0.920 | 0.898 | 0.532 m/s | 12/20 |
| Koda's best, as first measured | 0.1468 | 0.638 | 0.914 | 0.508 m/s | — |
| Best first-attempt reward, of 16 | 0.0380 | 1.000 | 0.774 | 0.644 m/s | — |

**The hand-written controller wins by 2.3x.** That is the headline and it is not buried.

Watch both:

```bash
./.venv/bin/python watch_policy.py \
    --policy runs/r3c1_zoohp/policy.zip \
    --reward results/search_12/candidates/round3/r3c1.py --speed 0.6
./.venv/bin/python sweep_mpc_baseline.py --speed 0.6
```

### Finding 1 — the learner was the bottleneck, not the reward search

`gamma`, PPO's discount factor, sets how far ahead the agent optimises. At 200 Hz:

```
gamma 0.99   (SB3 default)   ~100 control steps  = 0.5 s
gamma 0.999  (rl_zoo3)      ~1000 control steps  = 5.0 s
```

An Upkie is non-minimum phase: **to drive forward it must first drive its wheels backward**
for about half a second to tip the body over. At `gamma 0.99` that never pays off inside
the horizon the agent can see, so it learns to stay upright and not drive.

Same reward, same seed, only the PPO settings changing:

| | score | survives |
|---|---|---|
| 500k steps, SB3 defaults | 0.1468 | 0.638 |
| 1M steps, SB3 defaults | 0.1617 | 0.622 |
| 500k steps, rl_zoo3 settings | 0.1981 | 0.765 |
| **1M steps, rl_zoo3 settings** | **0.2286** | **0.920** |

**+56%, with no change to any reward function.** Doubling training under the bad discount
factor did nothing — survival even fell. Every reward Koda searched had been judged by that
learner, so every ranking it produced measured the reward and the learner together.

Settings taken from [`rl_zoo3_balancer`](https://codeberg.org/upkie/rl_zoo3_balancer), a
third-party SB3 balancer for this same robot. `tests/test_trainer_settings.py` checks each
value against it, and checks the units: 0.999 is 1000 control steps, which is 5.0 s at
200 Hz, not 5 steps and not 5 ms.

### Finding 2 — the model builds, it does not polish

Four times the model was handed its own best reward and asked to improve on it:

```
search_13 round 1   0.0245
search_14 round 1   0.0566
search_15 round 1   0.0179     <- learner fixed, feedback honest
                    bar: 0.1981
```

It rewrites rather than refines, and rewrites are far worse. Fixing the learner did not
change this.

What the loop *does* do is climb from wherever it is. In `search_15`, with everything
working: **0.0179 → 0.0358 → 0.0388**, monotonic for the first time. And `r3c1` — the best
reward in this repo — is itself a round-3 reward, reached by climbing from a cold start. Of
16 first-attempt rewards retrained under the fixed learner, the best scores **0.0380**,
five times worse.

So the iteration is where the value is, and three rounds is not many. **Eureka runs 400
candidates across 25 iterations and averages 5 training runs per reward. Koda ran 18 across
3, scored once each.** That ratio, not the prompt and not the model, is the difference we
can actually point at.

### Finding 3 — a grader that only counts falls will reward thrashing and standing still

Koda's first grader added survival, speed and distance together. Under it the top candidate
sat **pinned at the wheel torque limit for 96% of every control step** — it balanced by
vibrating. In the window it visibly convulses.

The grader now multiplies two gates in:

```
score = stayed_upright x torque_headroom
        x speed_accuracy^0.55 x distance_covered^0.45
        x straightness
```

A weighted **geometric** mean rather than an average, so **zero distance cannot be bought
back with anything else**, and by AM-GM it can never score higher than the average it
replaced. Re-scored under it, the old winner fell from 0.3101 to 0.0026 and placed twelfth
of seventeen. In the final search two candidates still appeared with survival 1.000 and
headroom 0.995 that travel −0.006 m. They score exactly 0.

---

## The bill

Published in full, failed runs included.

| run | calls | prompt tokens | completion tokens | CPU-hours |
|---|---|---|---|---|
| search_11 | 3 | 5,078 | 8,846 | 6.57 |
| search_12 | 3 | 5,157 | 10,598 | 6.99 |
| search_13 *(aborted — our bug)* | 2 | 1,758 | 4,070 | 1.83 |
| search_14 | 3 | 5,373 | 13,161 | 5.05 |
| search_15 | 3 | 5,319 | 12,512 | 5.43 |
| 16 first-attempt rewards retrained | 0 | — | — | ~6.0 |
| ablations and baselines | 0 | — | — | ~2.1 |
| **total** | **14** | **22,685** | **49,187** | **~34** |

```
model     nvidia/nemotron-3-super-120b-a12b, via Nebius Token Factory
machine   AMD Ryzen 7 5800H, 8 cores / 16 threads, 27 GB RAM
GPU       none
```

**`search_13`'s `run.json` records 1 call; it actually made 2.** The record is written at
the end of each round and round 2's call had already fired when the run was killed. Said
here because the file under-reports it.

**Two of the 14 calls were wasted on our own bug.** `--from-reward` set the starting reward
without its score, so round 1's winner replaced it whatever it scored. Caught mid-run from
the prompt contents, killed, fixed, and covered by a regression test.

---

## What is not claimed

- **No searched reward beat the hand-written controller.** 0.2286 against 0.5341.
- **One seed, one training run per number.** Eureka averages five. Everything here is
  single-sample, and the round-to-round noise is visible in the results.
- **Three rounds is not a trend.** `search_12` went 0.0740 → 0.0560 → 0.1468: a worse
  parent produced a better child, and a better parent later produced worse children.
- **Nothing ran on hardware.** No Upkie was involved.
- **The first search is retracted** and says so in its own file. It does not reproduce
  under the code that produced it and we never found out why.
- **Nine defects were found in our own measurement code** during this work, including a
  dead random seed, a leaked force between episodes, and a test suite that made paid API
  calls. All nine are listed with the commit that closed each in [`DEFECTS.md`](DEFECTS.md).

---

## Honesty notes

- **The human baseline is self-authored.** `human_baseline.py` is the entrant's own
  hand-written reward, committed and tagged `human-baseline` before any search was run.
  It is disclosed as self-authored because a baseline written after seeing model output
  would be a strawman.
- **Memorisation exposure is measured, not asserted.** Upkie reward functions are public
  on GitHub and Codeberg. The structural overlap between Nemotron's output and the
  published reward is reported as a number. The same caveat applies to the published
  Eureka results.
- **Deployability is architecture, not achievement.** Koda's output targets the same
  software spine that drives real Upkies. We do not own one and have not run anything on
  hardware.
- **The grader has deliberate blind spots.** It is complete on task achievement and
  hardware safety and silent on motion smoothness. Mirroring the regularisation terms a
  human would write would encode a human's answer in the grader and void the comparison.
  Blind spots found afterwards are the finding, not a bug.

---

## Prior art

| Work | Relationship |
|---|---|
| [Upkie](https://github.com/upkie/upkie) (Apache-2.0) | The robot, the simulation environment, and the hand-engineered PD / MPC / torque balancing controllers used as baselines |
| [rl_zoo3_balancer](https://codeberg.org/upkie/rl_zoo3_balancer) | An existing SB3 / PPO Upkie balancer. Third-party, RL, like-for-like — the strongest comparator available |
| [Bimo](https://github.com/mekion/the-bimo-project) | Printable biped with working sim-to-real |
| Eureka: Human-Level Reward Design via Coding Large Language Models (ICLR 2024) | The method being reproduced |

---

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Upkie's URDFs are
Apache-2.0 and Koda derives from them. The `upkie_description` mesh
`meshes/wheel_tire` is CC BY 4.0 and is attributed separately in `NOTICE`.
