# Koda — interface contract

**rev 4** · written 2026-10-18 · supersedes `../koda/INTERFACE.md` rev 3

Rev 3 was written for a quadruped on Isaac Lab. This project is a wheeled biped
(Upkie) on PyBullet. Rev 4 is the version the code in this repository actually
implements. What changed is listed at the bottom.

One sentence on why this file exists: two people and three programs have to agree
on the same shapes, so the shapes are written down in one place and each program
points at it.

---

## 1. Who may see what

```
  reward_terms.py ──▶ TrainerResult
  (the model wrote it)   │
                         ├── .terms   ──▶ reflect.py ──▶ the model, next round
                         └── .rollout ──▶ scorer.py
```

**`scorer.py` is handed `TrainerResult["rollout"]` and nothing else.** `rollout`
holds physical measurements. It contains no reward values, no reward term names,
and no score. That is not a rule someone has to remember — it is the only
argument the function takes.

`reflect.py` is the only place in Koda where grade *values* reach the model. Grade
*code* never does.

### The rule that decides what the grader may measure

| Kind of reward term | Example | May the grader measure it? |
|---|---|---|
| **Regularisation** — a penalty on *how* the motion is produced | action rate, joint acceleration, vertical velocity, torque magnitude | **Never.** Mirroring these puts a human's answer inside the grader, and three rounds of feedback then steer the model toward it. The comparison would collapse into "did the model rediscover our term list" |
| **Task achievement** — *was the commanded goal met* | speed tracking, ground covered, heading held | **Must.** A grader that does not measure the task measures nothing |
| **Hardware safety** — *would this break the robot* | peak torque against the motor's rated limit | **Must**, and as a threshold that is either crossed or not. Never as a continuous penalty |

The deliberate blind spots are action rate, jerk, vibration and energy
magnitude — anything about motion smoothness. They are preserved, not fixed.

---

## 2a. The reward function — what the model writes

**This is the signature. It is new in rev 4.** `human_baseline.py` and every
machine-written candidate implement exactly this and nothing else.

```python
def compute_reward(state: dict, action: dict) -> dict:
    """Called once per control step. Return your reward terms by name."""
```

**In:** two flat dictionaries of plain Python floats. Nested objects, numpy
arrays and simulator handles are deliberately absent, so the same call works
under the stub trainer and under PyBullet.

**Out:** a dict mapping your own term names to plain floats. The caller adds the
values together to get the step's reward. A term meant to discourage something
returns a negative number.

### Rules, all of them checked by Gate 1 before anything is trained

| Rule | What happens if it is broken |
|---|---|
| the file parses | `dead_compile` |
| it defines `compute_reward` taking exactly two positional arguments | `dead_gate1` |
| it returns a non-empty dict of `str` to number | `dead_gate1` |
| every value is finite — never NaN, never infinity | `dead_nan` |
| the same set of term names on every call | `dead_gate1` |
| only the standard library, and only `math` | `dead_gate1` |
| no file access, no sockets, no subprocesses | `dead_gate1` |

Term names are the author's own choice. **No term name is hard-coded anywhere in
Koda.** If one ever appears in `scorer.py`, the boundary in §1 has broken.

### `state` — 28 keys

Defined in code in `koda/state.py`. That file is the single source of truth; this
table is a copy for reading.

| Key | Unit | Meaning |
|---|---|---|
| `dt` | s | length of one control step (0.01 s, a 100 Hz loop) |
| `pitch` | rad | base pitch from vertical, positive leaning forward |
| `pitch_rate` | rad/s | base pitch angular velocity |
| `ground_position` | m | position of the average wheel contact point |
| `ground_velocity` | m/s | velocity of the average wheel contact point |
| `yaw` | rad | base yaw, zero at the start of the episode |
| `yaw_rate` | rad/s | base yaw angular velocity |
| `commanded_speed` | m/s | the forward speed the robot was asked for |
| `commanded_yaw_rate` | rad/s | the yaw rate it was asked for — zero in every evaluation episode |
| `contact` | 0 or 1 | 1.0 when at least one wheel touches the floor |
| `{joint}_position` | rad | joint angle |
| `{joint}_velocity` | rad/s | joint angular velocity |
| `{joint}_torque` | Nm | torque applied by that joint's motor |

`{joint}` is each of `left_hip`, `left_knee`, `left_wheel`, `right_hip`,
`right_knee`, `right_wheel`. Six joints, three fields each, plus the ten above.

### `action` — 2 keys

| Key | Unit | Meaning |
|---|---|---|
| `ground_velocity` | m/s | commanded forward velocity of the wheel contact point |
| `yaw_velocity` | rad/s | commanded yaw velocity |

These are the two numbers the policy emits. They go into `UpkieGyropod`, which
turns them into wheel velocity commands.

### A complete, working example

```python
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    return {
        "upright": math.exp(-8.0 * state["pitch"] ** 2),
        "speed_tracking": math.exp(-4.0 * speed_error * speed_error),
        "alive": 0.5 * state["contact"],
    }
```

Check any reward function against Gate 1 before committing it:

```
python3 -c "import sys; sys.path.insert(0,'.'); from koda import gate1; \
print(gate1.check(open('human_baseline.py').read())['status'])"
```

### What the model is told, and what it is not

The model is given this signature, the `state` and `action` tables above, the
goal sentence, and the environment's own `step` function showing the missing
reward. It is **not** given the motors' rated torque limits, because the grader
thresholds on those numbers. It is not given the grader, in any form.

The prompt is run through `koda/sanitize.py` first, which removes the robot's
name, its author's name and its motor part numbers. See the file for why.

---

## 2. `TrainerResult` — what every trainer returns, per candidate

Identical shape from `trainer_stub.py` and `trainer_pybullet.py`. The
orchestrator cannot tell them apart.

```json
{
  "schema_version": "2.0",
  "candidate_id": "r1c3",
  "round": 1,
  "trainer": "stub",
  "status": "trained",
  "error": null,
  "terms": {
    "checkpoints": [0.10, 0.25, 0.50, 1.00],
    "values": { "upright": [-1.60, -1.29, 0.83, 0.89],
                "speed_tracking": [-0.55, -0.51, -0.12, -0.12] },
    "total": [-2.15, -1.80, 0.71, 0.77]
  },
  "rollout": { "...see section 3..." },
  "policy_path": "runs/r1c3/policy.zip",
  "video_path": null,
  "train_seconds": 412.3,
  "compute_seconds": 412.3
}
```

| Field | Notes |
|---|---|
| `schema_version` | mandatory on every payload. Stub and real trainer must emit the same string or the orchestrator aborts. That is what stops them silently diverging |
| `status` | `trained` · `dead_compile` · `dead_gate1` · `dead_nan` · `dead_gate2` |
| `error` | the full traceback when `status` starts with `dead_`, otherwise `null`. **Goes back to the model verbatim** — it is how the model fixes its own bug, and it is pathology classes 1 and 2 |
| `terms.values` | keys are the model's own names. A term reading the same number at every checkpoint is a finding, and `reflect.py` says so in words |
| `terms.checkpoints` | fractions of training. Fixed at 10 / 25 / 50 / 100% |
| `compute_seconds` | **renamed from `gpu_seconds` in rev 3.** No GPU is rented, so the bill is CPU seconds |

A dead candidate still returns a `TrainerResult`, with `terms` and `rollout` set
to `null` and the traceback in `error`. Dead candidates are never dropped: they
are the pathology catalogue and they feed the next prompt.

---

## 3. `rollout` — physical facts only, the grader's sole input

```json
{
  "schema_version": "2.0",
  "episodes": [
    {
      "kind": "task",
      "seed": 11,
      "commanded_speed": 0.3,
      "commanded_yaw_rate": 0.0,
      "episode_seconds": 20.0,
      "steps_survived": 2000,
      "step_limit": 2000,
      "mean_speed_error": 0.041,
      "net_displacement_m": 5.68,
      "heading_error_deg": 4.1,
      "max_pitch_excursion_deg": 7.2,
      "closest_joint_margin_deg": 7.4,
      "peak_torque_nm": { "leg": 3.42, "wheel": 1.07 },
      "rms_torque_nm": { "leg": 1.73, "wheel": 0.41 }
    },
    {
      "kind": "push",
      "seed": 11,
      "commanded_speed": 0.6,
      "...": "every field above, plus:",
      "recovered": true,
      "recovery_time_s": 1.31
    }
  ]
}
```

### The field delta from rev 3

| rev 3 field | rev 4 |
|---|---|
| `steps_survived`, `step_limit`, `mean_speed_error`, `net_displacement_m`, `heading_error_deg`, `closest_joint_margin_deg` | unchanged |
| `max_foot_air_time_s` | **gone.** Upkie has no feet |
| `leg_effort_ratio` | **gone.** Two legs ending in wheels |
| `max_pitch_excursion_deg` | **new.** Worst pitch reached during the episode |
| `recovered`, `recovery_time_s` | **new.** Push episodes only |
| `peak_torque_nm`, `rms_torque_nm`, each split into `leg` and `wheel` | **new.** Raw newton-metres, per actuator class |

### Three things that are easy to get wrong

**`episode_seconds` is the actual elapsed simulated time** until the robot fell or
the step limit arrived, whichever came first. Never the step limit. Using the
limit would punish a fall twice — once through survival, and again through a
distance denominator that assumed a full episode.

**Torque is reported in newton-metres, not as a fraction of the limit.** `rollout`
holds raw physical facts; dividing by the rated limit is the grader's job. The
limits, from `upkie_description/urdf/upkie.urdf`:

| Joints | Motor | URDF `effort` |
|---|---|---|
| hip, knee (×4) | mjbots qdd100 | **16.0 Nm** |
| wheel (×2) | mjbots mj5208 | **1.7 Nm** |

**`recovery_time_s` when the robot never recovers** is set to the episode cap and
`recovered` is `false`. Averages are taken over recovered episodes only, and the
number of non-recoveries is reported beside the average, never averaged in as
though it were a slow recovery.

### Fields that must never be added

| Tempting field | What it mirrors | Verdict |
|---|---|---|
| action rate, joint acceleration | regularisation | **never** — jitter stays a blind spot |
| vertical velocity | regularisation | **never** — hopping stays a blind spot |
| any total-energy or summed-torque number | regularisation | **never** — a threshold on peak torque is safety; a magnitude penalty is regularisation |

---

## 4. `ScoreResult` — what the grader returns

```json
{
  "schema_version": "2.0",
  "candidate_id": "r1c3",
  "scorer": "stub",
  "score": 0.742,
  "per_speed": { "0.3": 0.81, "0.6": 0.76, "0.9": 0.66 },
  "measurements": {
    "survival_frac": 1.0,
    "mean_speed_error": 0.052,
    "displacement_frac": 0.94,
    "heading_error_deg": 6.8,
    "max_pitch_excursion_deg": 11.2,
    "closest_joint_margin_deg": 6.1,
    "peak_torque_frac_leg": 0.43,
    "peak_torque_frac_wheel": 0.74,
    "recovery_time_s": 1.26,
    "non_recovery_count": 1
  },
  "active_measurements": ["survival_frac", "mean_speed_error",
                          "displacement_frac", "heading_error_deg"],
  "violations": ["wheel_torque_exceeded"],
  "scorer_sha": "3f9a1c7"
}
```

### Reducers across the 20 episodes — three of them are not means

| Field | Reduced by | Why |
|---|---|---|
| `survival_frac` | mean of `steps_survived / step_limit` | |
| `mean_speed_error` | mean | |
| `displacement_frac` | mean of `net_displacement_m / (commanded_speed × episode_seconds)` | derived by the grader. Speed-invariant, so it adds up across the three speeds. **Negative is kept, not clamped** — going backwards is informative |
| `heading_error_deg` | **max** | worst drift |
| `max_pitch_excursion_deg` | **max** | worst case |
| `closest_joint_margin_deg` | **min** | a mean would hide one near-miss |
| `peak_torque_frac_*` | **max**, then divided by the rated limit | anything above 1.0 is over the limit |
| `recovery_time_s` | mean over recovered episodes only | |

`displacement_frac` overlaps `mean_speed_error` on purpose: a policy translating
in a tight circle scores well on one and badly on the other. The grader weights
them so forward progress is not counted twice.

`active_measurements` is the list the score is computed over, pre-registered at
`prereg-v1`. **Scores from searches with different `active_measurements` are never
compared as though they used the same rule.**

`scorer_sha` is the sha of the grader **as actually run**, not the pre-registered
value — otherwise it could not detect the mismatch that makes it useful. A sha
that differs from the `prereg-v1` one *is* the disclosure.

`violations[]` is a closed list of strings, owned by the other pane as part of the
grader. The stand-in grader currently emits `wheel_torque_exceeded`,
`leg_torque_exceeded`, `joint_limit_hit` and `never_recovered_from_push`.

---

## 5. The evaluation episodes

In code in `koda/eval_set.py`. Frozen at `git tag prereg-v1`.

**20 episodes per candidate:**

| Count | Kind | Detail |
|---|---|---|
| 15 | task | 5 fixed seeds (11, 23, 37, 53, 71) × 3 commanded speeds (0.3, 0.6, 0.9 m/s) |
| 5 | push | the same 5 seeds at 0.6 m/s, one pre-registered shove each |

The shove: 25 N forward for 0.1 s, applied 5 s into the episode. Recovery means
pitch back inside 0.15 rad and held there for 1 s. Step limit 2000, which is 20 s
at 100 Hz. No exploration noise in any episode. Commanded yaw rate is zero
everywhere, so any heading drift is the policy's own doing.

> **Open, needs the user's decision before `prereg-v1`:** rev 3 pinned 15
> episodes and said nothing about push episodes. The 5 push episodes, the 25 N
> magnitude and the 5 s timing above are a proposal, not a settled number. They
> are cheap to change now and frozen on Oct 23.

---

## 6. The run record

One JSON file per search, written after every round, so a search that dies
halfway still renders a report. Shape: see `out/run.json` from any run.

```
schema_version · search_id · robot · env · goal
config          candidates, rounds, seed, trainer, scorer, candidate_source,
                episodes_per_candidate
human_baseline  the entrant's own reward, run through the identical path, with
                self_authored: true and the disclosure sentence
rounds[]        round, prompt_sha, prompt_path, candidates[], winner, reflection,
                survivors, died
pathologies[]   class, round, candidate_id, label, detail, passed_scorer
bill            compute_seconds, usd_compute, usd_tokens
```

`pathologies[]` classes:

| Class | What it is | Where it comes from |
|---|---|---|
| 1 | did not compile, or failed Gate 1 | automatic, from `status` |
| 2 | the reward or the training went non-finite | automatic, from `status` |
| 3 | scored on the task and still broke a hardware limit | automatic, from `violations` |
| 4 | scores well and is visibly wrong | **by hand, from the evaluation video.** Absent from any run with no video, rather than guessed at |

---

## 7. Who owns what

| Who | Files |
|---|---|
| `master` | `orchestrator.py`, `prompt.py`, `sanitize.py`, `gate1.py`, `candidates.py`, `state.py`, `eval_set.py`, `trainer_stub.py`, `trainer_pybullet.py`, `reflect.py`, `report.py`, `cli.py`, `fixtures/`, this file |
| `planner` | `scorer.py`, the three cheating rewards in `cheaters/`, the `violations[]` list |
| the user | `human_baseline.py` |

`koda/scorer_stub.py` is a stand-in so the loop could be finished before the real
grader exists. It is not the project's grader and no scored run may use it.
Swapping the real one in is one import line in `koda/orchestrator.py`.

---

## Changelog

**rev 4** — Rewritten for Upkie on PyBullet. **Added §2a, the reward-function
signature, which did not exist in any form before.** Applied the `rollout` field
delta: `max_foot_air_time_s` and `leg_effort_ratio` deleted,
`max_pitch_excursion_deg`, `recovered`, `recovery_time_s`, `peak_torque_nm` and
`rms_torque_nm` added. Torque reported raw in newton-metres instead of as a
fraction. `survival_frac` chosen as the single name for what rev 3 called
`survival_frac` and the plan called `upright_frac`; it is the fraction of the
episode the robot stayed up. Episode count stated: 20, as 15 task plus 5 push.
`gpu_seconds` renamed `compute_seconds`, because no GPU is rented.
`schema_version` bumped to 2.0. Pathology class 3 defined so it can be tagged
automatically. `active_measurements` kept; the Search B exclusion of
`leg_effort_ratio` is gone with the field.

**rev 3** — `episode_seconds` pinned to actual elapsed time. Negative
`displacement_frac` kept unclamped. Contact sheet is triage only.

**rev 2** — Added `net_displacement_m` and `heading_error_deg`. Zero commanded
yaw rate at evaluation. Reducers pinned. `active_measurements` added.
`scorer_sha` is the sha as run.
