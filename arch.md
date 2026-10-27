# How to read this code

Written 2026-10-26, for someone opening `koda_v2` for the first time. Describes
**Stage A**: the loop running end to end on hand-written fixtures, with no API
call, no simulator and no money spent.

Every `file:line` below is real. Jump straight to it.

---

## Part 1 — the algorithm

### Level 1: one screen

```
                        ┌──────────────────────────────────┐
                        │  IN:  a robot, one sentence      │
                        │       "balance upright and       │
                        │        drive forward at the      │
                        │        commanded speed"          │
                        └────────────────┬─────────────────┘
                                         │
                          ╔══════════════▼═══════════════╗
                          ║   round = 1                  ║
                          ║   best    = nothing yet      ║
                          ║   lessons = nothing yet      ║
                          ╚══════════════╤═══════════════╝
                                         │
    ┌────────────────────────────────────▼─────────────────────────────────┐
    │                                                                      │
    │   1. WRITE THE QUESTION      the env's own `reward = 0.0` line        │
    │                            + the goal sentence                       │
    │                            + the 28 numbers a reward may read        │
    │                            + best and lessons, if any                │
    │                                      │                               │
    │   2. HIDE WHO THE ROBOT IS           ▼   remove "Upkie", "qdd100",   │
    │                                          "mj5208", the author's name │
    │                                      │                               │
    │   3. ASK FOR SIX ANSWERS             ▼   six reward functions        │
    │                                          (Stage A: read off disk)    │
    │                                      │                               │
    │   4. CHECK EACH ONE, CHEAPLY         ▼   does it parse? does it run? │
    │                                          finite numbers? — see L2    │
    │                                      │                               │
    │   5. TRAIN EACH SURVIVOR             ▼   (Stage A: made-up physics)  │
    │                                      │                               │
    │   6. GRADE EACH TRAINED ROBOT        ▼   physical measurements only  │
    │                                      │                               │
    │   7. PICK THE BEST, WRITE DOWN       ▼   best    = its source code   │
    │      WHAT HAPPENED                       lessons = plain English     │
    │                                      │                               │
    │   8. SAVE EVERYTHING TO DISK         ▼   out/run.json                │
    │                                      │                               │
    └──────────────────────────────────────┼───────────────────────────────┘
                                           │
                              ╱────────────▼────────────╲
                             ╱   round < 3 ?             ╲
                             ╲                           ╱
                              ╲────┬──────────────┬─────╱
                            yes│              no  │
                               │                  │
             round = round + 1 │                  ▼
             ──────────────────┘        ┌─────────────────────────┐
             go back to step 1          │  OUT: out/run.json      │
             carrying best + lessons    │       out/report.html   │
                                        │       out/prompts/*.txt │
                                        └─────────────────────────┘
```

**The only thing that makes this a loop and not six independent guesses** is the
arrow at the bottom carrying `best` and `lessons` back up to step 1.

---

### Level 2: what happens to one candidate

Six candidates go through this, independently, every round. This is the part with
real decisions in it.

```
                    one candidate: a string of Python source
                                     │
                                     ▼
                       ╱─────────────────────────╲
                      ╱   does the file parse?    ╲──── no ──┐
                      ╲                           ╱          │
                       ╲──────────┬──────────────╱           │
                             yes  │                          ▼
                                  ▼                    dead_compile
            ╱─────────────────────────────────────╲          │
           ╱  def compute_reward(state, action) ?  ╲── no ───┤
           ╲  only `import math` ?                 ╱         │
            ╲  no open(), exec(), __dunder__ ?    ╱          ▼
             ╲───────────────┬────────────────────╱     dead_gate1
                       yes   │                               │
                             ▼                               │
                 call it on 5 made-up states:                │
                 still · moving · fallen · zero cmd · at limit
                             │                               │
            ╱────────────────▼──────────────────╲            │
           ╱   did it raise?                     ╲── yes ────┤
           ╱    return a non-empty dict?          ╲── no ────┤
           ╲    same term names every call?       ╱── no ────┤
            ╲──────────────┬─────────────────────╱           │
                     all ok │                                │
                            ▼                                │
            ╱───────────────────────────────────╲             ▼
           ╱  every value finite?                ╲── no ─ dead_nan
           ╲  (no NaN, no infinity)              ╱            │
            ╲──────────────┬────────────────────╱             │
                      yes  │                                  │
          ══════════════════▼═══════════════════               │
          ║  PASSED GATE 1                     ║               │
          ║  cost so far: a few milliseconds   ║               │
          ══════════════════╤═══════════════════               │
                            ▼                                  │
                        TRAIN IT                                │
                     (the expensive step)                       │
                            │                                   │
            ╱───────────────▼───────────────────╲                │
           ╱  did training stay finite?          ╲── no ─ dead_gate2
           ╲                                     ╱              │
            ╲──────────────┬────────────────────╱               │
                      yes  │                                    │
                           ▼                                    │
              ┌────────────────────────┐                         │
              │ terms   ──► the lessons│                         │
              │ rollout ──► the grader │                         │
              └───────────┬────────────┘                         │
                          ▼                                      │
                       GRADE IT                                  │
                     score 0.000 to 1.000                        │
                     + limits exceeded?                          │
                          │                                      │
                          ▼                                      ▼
              ┌───────────────────────┐        ┌──────────────────────────┐
              │ a number to rank by   │        │ NOT thrown away.         │
              │                       │        │ The error text goes into │
              │                       │        │ the next prompt, word    │
              │                       │        │ for word, and into the   │
              │                       │        │ pathology catalogue.     │
              └───────────────────────┘        └──────────────────────────┘
```

**Read the right-hand column.** A dead candidate is not a failure the system
hides — it is output. That is why `fixtures/` has four broken files in it on
purpose.

---

### Level 3: what crosses between the rounds

The two arrows that matter, and the wall between them.

```
   ROUND 1                                                      ROUND 2
   ═══════                                                      ═══════

   prompt (4,701 chars)
   the env's reward hole
   + the goal
   + the 28 state keys
          │
          ▼
   6 candidates
          │
          ├── r1c2  dead_compile ──── error text ────────┐
          ├── r1c3  dead_nan     ──── error text ────────┤
          ├── r1c4  dead_gate1   ──── error text ────────┤
          │                                              │
          ├── r1c5  0.252                                │
          ├── r1c6  0.048                                │
          └── r1c1  0.307  ◄── WINNER                    │
                     │                                   │
          ┌──────────┴──────────┐                        │
          ▼                     ▼                        │
       terms                 rollout                     │
   per-term curves      physical numbers                 │
          │                     │                        │
          │                     ▼                        │
          │              ┌─────────────┐                 │
          │              │ THE GRADER  │                 │
          │              │             │                 │
          │              │ never sees  │                 │
          │              │ terms, term │                 │
          │              │ names, or   │                 │
          │              │ the reward  │                 │
          │              └──────┬──────┘                 │
          │                     │                        │
          │                measurements                  │
          │                     │                        │
          └─────────┬───────────┘                        │
                    ▼                                    │
            ┌───────────────┐                            │
            │  reflect.py   │                            │
            │               │                            │
            │ the ONLY place│                            │
            │ grade numbers │                            │
            │ reach the     │                            │
            │ model         │                            │
            └───────┬───────┘                            │
                    │                                    │
                 lessons ───────────────┐                │
          r1c1's source ────────────────┤                │
                                        │                │
                                        ▼                ▼
                               prompt (6,775 chars) ◄────┘
                               ─────────────────────
                               everything from round 1
                               + r1c1's source code
                               + the lessons
                               + 3 error messages
                                        │
                                        ▼
                                  6 candidates
                                  best: 0.479   ◄── up from 0.307
```

The prompt grew from 4,701 to 6,775 characters. That 2,074-character difference
**is** the feedback.

---

### Level 4: the three rounds that actually ran

One chart. Three **columns**, not three plots. The x-axis is the round, the
y-axis is the score. Each column holds that round's 6 candidates — a dot if it
scored, an ✗ if it died.

```
  score
  0.60 ┤                                        ● 0.599
       │                                        ● 0.585
       │                                        ● 0.584
       │                                        ● 0.581
  0.48 ┤                   ● 0.479
       │                   ● 0.430
  0.36 ┤                   ● 0.358
       │                   ● 0.347
  0.31 ┤  ● 0.307          ● 0.339
       │  ● 0.252
  0.05 ┤  ● 0.048
       │
  dead ┤  ✗ ✗ ✗            ✗                    ✗ ✗
       └───────────────────────────────────────────────
         round 1           round 2              round 3
         6 candidates      6 candidates         6 candidates
         3 scored, 3 died  5 scored, 1 died     4 scored, 2 died
```

18 candidates total: 6 per round × 3 rounds. 12 scored, 6 died.

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| best score | 0.307 | 0.479 | **0.599** |
| survived | 3 of 6 | 5 of 6 | 4 of 6 |
| died in Gate 1 | 3 | 1 | 0 |
| died in training | 0 | 0 | 2 |
| prompt size | 4,701 | 6,775 | 6,658 |

**Round 3 killed two candidates that round 1 would not have.** Gate 1 got nothing
in round 3 — every file parsed and ran. The two deaths were the stub's simulated
training divergence, which is the failure Gate 1 cannot see by design: you only
find it by spending the compute.

### The one honest caveat on that chart

The rising line is **not** evidence the method works. The round 2 and round 3
fixtures are files that were hand-written to be better than the round 1 ones, and
the stub trainer adds `+0.06` per round outright. In Stage A that curve is proof
the plumbing carries feedback forward, nothing more.

The same chart drawn from real Nemotron output and real PyBullet training would be
result **R2** in the plan. This one is a wiring diagram.

Reproduce it with:

```
python3 -m koda.cli run --robot upkie --goal "balance upright and drive forward at the commanded speed" --seed 1
```

---

## Part 2 — the call flow

### The whole command

```
you type:
  python3 -m koda.cli run --robot upkie --goal "balance upright and drive forward ..."
        │
        ▼
  cli.py:22   main()
        │     reads the flags, makes two calls, prints the summary
        │
        ├──► orchestrator.py:135   run(...)  ──► returns one big dict: the run record
        │                                        and writes it to out/run.json
        │
        └──► report.py:270         write(record, "out/report.html")
```

Two calls. That is the entire top level.

---

### Inside `run()` — one round, done three times

The loop body starts at `orchestrator.py:184`.

```
orchestrator.py:135  run()
  │
  │  once, only if human_baseline.py exists:
  ├── _human_baseline()  orchestrator.py:118 ──► _grade_one()   same path as any candidate
  │
  │  for round_number in 1, 2, 3:                orchestrator.py:184
  │
  │  ① BUILD THE PROMPT          prompt.py:76   build(goal, 6, best_source, reflection, errors)
  │        ├── prompt.py:26  env_excerpt()   the env's own `reward = 0.0` lines
  │        └── prompt.py:44  field_block() ──► state.py:72  field_table()   28 keys + units
  │     round 1: nothing extra.
  │     rounds 2 and 3: + the winner's source code
  │                     + the winner's reflection text
  │                     + every dead candidate's error, verbatim
  │
  │  ② STRIP THE IDENTITY        sanitize.py:75  sanitize_or_raise(text)
  │        ├── sanitize.py:60  sanitize()   ordered replacements, then a case-blind sweep
  │        └── sanitize.py:69  leaks()      anything survives ──► raise. Prompt never sent.
  │     then writes out/prompts/roundN.txt — exactly what would go out
  │
  │  ③ GET SIX CANDIDATES        candidates.py:17  from_fixtures(round_number, 6)
  │     ◄── THIS IS THE STUBBED STEP. Reads fixtures/roundN/*.py in filename order.
  │         Stage B: from_nemotron(prompt) goes here, returning the same list shape.
  │
  │  ④ FOR EACH OF THE SIX       orchestrator.py:99  _grade_one()   ── see next diagram
  │
  │  ⑤ PICK THE WINNER           highest score among the ones that trained
  │  ⑥ WRITE THE FEEDBACK        reflect.py:29   reflect(winner_trainer, winner_score)
  │  ⑦ LOG PATHOLOGIES           orchestrator.py:52  _pathologies()   classes 1, 2, 3
  │  ⑧ WRITE out/run.json        orchestrator.py:238  ── after EVERY round, not just the end
  │
  └── the winner's source and reflection are the inputs to ① next round
```

That last line is the loop. Everything else is bookkeeping.

---

### One candidate's journey — the part worth reviewing hardest

```
   candidate source text (just a string)
        │
        ▼
   gate1.py:144  check(source)
        │   ast.parse                               fails ──► "dead_compile"
        │   _check_source     gate1.py:84           fails ──► "dead_gate1"
        │     (signature, banned imports, banned names, dunder access)
        │   exec, in a namespace holding only `math` and 20 safe builtins
        │   then call it on 5 states    gate1.py:117  _test_states()
        │     standing still · mid-trajectory · fallen over · zero command · every limit
        │        not a dict / raised / term names changed ──► "dead_gate1"
        │        NaN or infinity                          ──► "dead_nan"
        ▼
   verdict = {"ok": True, "term_names": [...], "reward_fn": <the live function>}
        │
        │   if not ok ──► _dead_result()  orchestrator.py:34
        │                 TrainerResult with terms=null, rollout=null, error=traceback
        │                 training and grading are SKIPPED. It still goes in the report,
        │                 and its error text goes in the next prompt.
        ▼
   trainer_stub.py:181  train(reward_fn, candidate_id, round, goal, seed)
        │   _discrimination  trainer_stub.py:97   ── calls reward_fn on 3 trajectories
        │       prefers upright to fallen?        ──► "balance"
        │       prefers moving to sitting still?  ──► "motion"
        │   _term_curves     trainer_stub.py:56   ── calls reward_fn again
        │       ──► terms.values  {the model's own term names: [4 numbers]}      REAL
        │   _rollout         trainer_stub.py:116
        │       ──► 20 episodes of physical numbers                          MADE UP
        │
        │   3,200 calls to the candidate's reward function per train(). Counted.
        ▼
   TrainerResult = {status, error, terms, rollout, train_seconds, compute_seconds}
        │
        │   rollout only. terms do not cross this line.
        ▼
   scorer_stub.py:65  score(rollout, candidate_id)
        │   20 episodes ──► 10 measurements. Three are max or min, not means.
        │   peak torque ÷ 16.0 Nm and ÷ 1.7 Nm ──► violations[]
        ▼
   ScoreResult = {score, per_speed, measurements, violations, scorer_sha}
```

---

### The boundary the project rests on

```
                      ┌── terms   ──► reflect.py:29 ──► the next prompt ──► the model
    TrainerResult ────┤
                      └── rollout ──► scorer_stub.py:65  score(rollout, candidate_id)
                                                          ▲
                                                          │ that dict is the function's
                                                          │ only data argument. No reward
                                                          │ values, no term names, no score.
```

`reflect.py` is the one place grade *numbers* reach the model. Grade *code* never
does. `tests/test_loop.py` greps `score()`'s own source for the words `terms` and
`reward` and fails if either appears.

Why it matters: the grader exists to judge machine-written rewards from the
outside. If a reward term name could reach it, the grader would be scoring the
model's own vocabulary back at it, and the comparison the whole project rests on
would be void. `INTERFACE.md` section 1 has the full rule.

---

## Part 3 — read the files in this order

| # | File | Lines | Why this one next |
|---|---|---|---|
| 1 | `koda/state.py` | 212 | The reward signature, in code. Everything downstream is shaped by the 28 keys here |
| 2 | `koda/gate1.py` | 215 | Where machine-written code is first executed. The place to be suspicious |
| 3 | `koda/trainer_stub.py` | 236 | Read `_discrimination` and `_rollout`. This is where real ends and made-up begins |
| 4 | `koda/scorer_stub.py` | 132 | Short. Check it only ever touches `rollout` |
| 5 | `koda/orchestrator.py` | 252 | The loop itself. Lines 184–250 are the eight steps above |
| 6 | `koda/prompt.py` + `koda/sanitize.py` | 134 + 84 | What actually leaves the machine. Compare against `out/prompts/round2.txt` |
| 7 | `koda/report.py` | 281 | Formatting only. No logic. Skim it |

`candidates.py` (44), `eval_set.py` (58), `cli.py` (80) and `reflect.py` (117) are
small and have no surprises in them.

---

## Part 4 — the two swap points for Stage B

`koda/orchestrator.py:30-31`, and nothing else changes:

```python
from koda import scorer_stub as scorer      # ← becomes `scorer`  (the other pane's file)
from koda import trainer_stub as trainer    # ← becomes `trainer_pybullet`
```

Both modules expose the same small surface:

| Module | Must provide |
|---|---|
| trainer | `name()` and `train(reward_fn, candidate_id, round_number, goal, seed)` → a TrainerResult |
| grader | `name()`, `scorer_sha()` and `score(rollout, candidate_id)` → a ScoreResult |

A third swap point arrives with the real model call: `candidates.from_fixtures()`
at `orchestrator.py:196` becomes `candidates.from_nemotron(prompt)`, returning the
same list of `{candidate_id, source_path, source}` dicts.

---

## Part 5 — where to push on it in review

1. **`gate1.py:42` `_safe_import`.** It lets `import math` through and blocks the
   rest. This is a guard rail, not a sandbox. Machine-written code still gets
   `exec`'d in this process.
2. **`trainer_stub.py:97` `_discrimination`.** The whole ranking in Stage A rests
   on two numbers: does the reward prefer upright to fallen, and does it prefer
   moving to sitting still. If those are the wrong two axes, say so now.
3. **`scorer_stub.py:46` `_task_score`.** The weights are 0.45 survival, 0.30
   speed, 0.25 distance, then multiplied by a heading factor. Those were invented
   for the stand-in. The real grader is the other pane's and will have its own.
4. **`state.py:90-91` the torque limits.** 16.0 Nm and 1.7 Nm live in `state.py`
   but are deliberately kept out of `prompt.py`, so the model is never told the
   numbers the grader thresholds on. Reversible.
5. **`eval_set.py:23-25` the push.** 25 N, 0.1 s, at 5 seconds in. Invented, and
   flagged in `INTERFACE.md` section 5 as needing the user's decision before
   `prereg-v1`.

---

## Two things this document will not dress up

**The grader is a stand-in.** There is no `scorer.py` in this repository.
`koda/scorer_stub.py` exists so the loop had something to call. Its own header
says it is not the project's grader and no scored run may use it.

**The stub trainer is not physics.** Its made-up numbers are steered by two
things it genuinely measures about each reward function. That is enough to rank
the "pay the robot to do nothing" fixture below the honest ones — the first
version used a single quality number and that fixture won round 1, which is how
the gap was found. But a cheating reward cannot really be caught until Stage B,
because in Stage A the thing being graded is arithmetic, not a robot.
