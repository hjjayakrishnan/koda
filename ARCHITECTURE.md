# How Koda works

One command, three rounds, three model calls.

```
koda run --run-id search_15 --goal "balance upright and drive forward at the commanded speed"
```

```
                        koda/cli.py
                             |
                             v
                   koda/orchestrator.py
                             |
   +=========================|=== ROUND n, for n = 1, 2, 3 ================+
   |                         v                                             |
   |   koda/prompt.py : build()                                            |
   |      the goal, one sentence                                           |
   |      Upkie's step() source, with its reward line removed              |
   |      the state and action fields, with units                          |
   |      the contract: def compute_reward(state, action) -> dict          |
   |      the best reward so far + what happened to it  <---+ FEEDBACK     |
   |                         |                              |  (round 2 on)|
   |                         v                              |              |
   |   koda/sanitize.py      strips the robot's name and part numbers      |
   |                         |                              |              |
   |                         v                              |              |
   |   koda/candidates.py    ---- HTTPS ---->  NEBIUS TOKEN FACTORY        |
   |                         <--------------   Nemotron 3 Super            |
   |                         |                 6 reward functions          |
   |                         v                              |              |
   |   koda/gate1.py         compiles? right signature?     |              |
   |         |               finite floats?                 |              |
   |         +--- no ------> recorded as dead, error text ->+              |
   |         | yes                                          |              |
   |         v                                              |              |
   |   koda/trainer_pybullet.py                             |              |
   |      PPO, 500,000 steps, 6 candidates in parallel      |              |
   |      gamma 0.999, lin_0.003, observation normalisation |              |
   |         |                                              |              |
   |         v                                              |              |
   |      20 scoring episodes, a FRESH environment each     |              |
   |      3 speeds x 5 starts + 5 shove tests               |              |
   |         |                                              |              |
   |         v                                              |              |
   |      rollout = raw physics only                        |              |
   |      steps survived, metres travelled, speed error,    |              |
   |      peak torque, steps spent pinned at the limit      |              |
   |         |                                              |              |
   |         v                                              |              |
   |   koda/scorer.py        ** the model NEVER sees this file **          |
   |      score = stayed_upright x torque_headroom                         |
   |              x speed_accuracy^0.55 x distance_covered^0.45            |
   |              x straightness                                           |
   |         |                                              |              |
   |         v                                              |              |
   |      best of the 6. Becomes the next parent ONLY if it                |
   |      beats every round before it.                      |              |
   |         |                                              |              |
   |         v                                              |              |
   |   koda/reflect.py       turns measurements into English               |
   |      "stayed up for 60% of the episode"                |              |
   |      "wheel motors sat pinned at that limit for 17%"   |              |
   |      "Score: 0.036 out of 1.000"                       |              |
   |         |                                              |              |
   |         +----------------------------------------------+              |
   +=======================================================================+
                             |
                             v
                runs/<run-id>/seed11/<candidate>/policy.zip
                out/<run-id>/seed11/run.json
```

See [`FEEDBACK_EXAMPLE.md`](FEEDBACK_EXAMPLE.md) for the real text that travels
along the feedback arrow.

## The four parts that matter

**The model writes code, not numbers.** It receives a function signature and
returns a Python file. It chooses its own reward terms and names them. Nothing
is templated.

**The grader is separate and hidden.** `koda/scorer.py` scores from physical
measurements alone — it cannot see reward values or term names, and a test
checks that against the source text rather than trusting discipline. The model
sees the measurements and the resulting score, never the formula.

**The grader has deliberate blind spots.** `tests/test_scorer.py` keeps a list —
`jerk`, `action_rate`, `energy`, `smoothness`, `vibration` — and asserts the
grader measures none of them, so the model has to discover those itself. It
does: one round-3 candidate wrote its own `jerk_penalty` without being prompted.

**The loop keeps its best, not its latest.** A round's winner becomes the next
parent only if it beats every earlier round. Without that rule one search bred
its third round from a parent worse than its first.

## Scale, stated plainly

| | Eureka | Koda |
|---|---|---|
| simulator | IsaacGym, GPU | PyBullet, CPU |
| candidates per robot | 400 | 18 |
| samples per iteration | 16 | 6 |
| iterations | 5, x5 restarts | 3 |
| training runs averaged per reward | 5 | 1 |
| hardware | GPUs | 16-thread laptop, no GPU |

That ratio is the difference we can point at. It is not the prompt and it is not
the model.
