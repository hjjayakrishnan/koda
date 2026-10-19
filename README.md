# Koda

**Give it a robot and one sentence. Get back the reward function and a trained robot.**

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

<!--
  INTERNAL NOTE — not rendered on GitHub, and intentionally not a visible TODO.
  Plan §9.8 forbids placeholder text in anything a judge sees, so every pending
  item in this file is an HTML comment. Find them all before the Oct 27 draft
  submission with:  grep -n 'PENDING(koda)' README.md
  None may survive to the final submission.
-->

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

All training executes on Nebius AI Cloud. The orchestrator and report generation run on
a laptop.

<!-- PENDING(koda): replace this comment with the measured cost once the runs are done.
     Plan §9.7 requires publishing the exact bill including failed runs. -->

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

<!-- PENDING(koda): the baseline ladder, the pathology catalogue, search-level
     reproducibility, damage adaptation, the second task, and the deployability latency
     number all land here. Plan §4 is the list and §9.9 requires reporting negatives
     with the number if Nemotron loses to any baseline. -->

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
