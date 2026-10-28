# Lessons learnt

What we found out, on the day we found it out. Newest at the bottom.

Rules for this file: only things that changed what we did. Each entry says what we
believed, what turned out to be true, and what it cost. No restating the plan.

---

## 2026-10-18

**The premise is a quote, not a claim.** Upkie's environments ship with `reward = 0.0`.
The maintainers wrote reward functions, then deliberately deleted them:
`upkie/envs/upkie_env.py` line 230 reads `reward = 0.0  # to be decided by agents`.
We had been describing the project as "filling a gap we identified." It is better than
that — the gap is documented upstream, in their changelog, and a judge can open the file.

**The reward function signature did not exist anywhere.** The plan listed `INTERFACE.md`
as "done". It defined `TrainerResult`, `rollout`, `ScoreResult` and the run JSON, and said
nothing about what a reward function is called with or returns — the one interface the
whole project is built on. It had to be invented, and it was blocking the baseline.
Cost: nothing, because it was caught by a review before anyone depended on it.

**Build the loop against fakes first.** Steps 1, 3, 5, 6, 7 and 8 were built and debugged
with hand-written reward files standing in for the model and a stub trainer standing in for
PPO. No API calls, no compute, deterministic. Everything found later was found in the real
pieces, not in the plumbing.

**Deliberately broken fixtures earn their keep.** `fixtures/` includes a syntax error, a NaN
return, a wrong tensor shape and a wrong signature. Those exercised Gate 1's error paths and
the traceback-back-to-the-model path before either mattered.

---

## 2026-10-26

**Run the vendor's own controller before trusting your stack.** `pd_balancing` is Upkie's,
unmodified. It kept the robot up for 5 of 5 episodes, worst pitch 0.91 deg. That separated
"our code is broken" from "the environment is broken" permanently, and cost ten minutes.

**The baseline we had chosen could not do the task.** The PD controller takes no speed
command — three gains reading current state, one of which pulls it back to where it started.
It holds still. Our eval set asks for 0.3, 0.6 and 0.9 m/s, so scoring it against that would
have been scoring it on a task it was never given. `displacement_frac` came back as
**-93300.0**, which is what a zero commanded speed does to a ratio. Switched to
`upkie.controllers.MPCBalancer`, which does take a speed target.

**Compiled dependencies were not the risk we thought.** `proxsuite` and `qpmpc` were flagged
as a Python 3.14 compilation risk. Both had prebuilt `cp314` wheels. Install took seconds.
The lesson is narrow: check for a wheel before budgeting for a build.

**The simulator sleeps by default.** `regulate_frequency` defaults to `True`, so a 20-second
episode takes 20 seconds of wall clock. Correct for driving a real robot, wrong for training.

    regulate_frequency=True    196 control steps/s
    regulate_frequency=False   549-664 control steps/s      3x faster

Over an 18-candidate search that is 45 minutes versus 2.5 hours. It is now tied to `gui`:
a window a person watches plays at real speed, a headless run goes flat out.

**A static camera is not a viewer.** The robot travels 11 m in 20 s and left the frame in
seconds. Rendering frames after the fact is not the same as watching it. The camera now
follows the robot.

---

## 2026-10-27

**The reward went up while the robot fell over every time.** This is the central finding so
far and it is the project's whole thesis appearing on day one. On a 500k-step run the reward
sum climbed 1.65 -> 2.12 out of a possible 2.5, and **0 of 20 evaluation episodes survived**.
Per-step reward stays high for as long as the episode lasts; nothing in that reward noticed
the falling. If `reflect.py` reports only the reward curve, the model is told "your reward
improved" about a policy that fails completely.

**`VecNormalize` is catastrophic here, and the control run is what proved it.** Observation
and reward normalisation was the hypothesis for why training was failing. It was backwards:

    200k steps, normalised     4.55 / 5.54 / 3.89 s survived
    200k steps, unnormalised   20.00 / 11.29 / 20.00 s
    1M steps,   normalised     1.52 / 1.54 / 1.28 s

It gets worse with more training. Without the unnormalised control it would have been
committed as a fix. Likely cause worth noting: `VecNormalize` rescales the **reward** as
well as the observations, and this project is specifically about reward functions.

**The measurement was corrupting the thing it measured.** To report per-term reward averages,
the trainer paused at each checkpoint and drove the **training environment** for 600 steps —
`reset()`, an overridden commanded speed, 600 of its own steps — then handed it back to PPO
mid-`learn()`. Same reward, same seed, same 500k budget:

    measuring on the training env     32.5 s survived in total
    reading the training rollouts     52.3 s survived in total

Fixed with an SB3 callback that reads `info["reward_terms"]` out of rollouts PPO already
produced. Nothing driven, nothing reset. General form of the lesson: if a diagnostic touches
the system it is diagnosing, assume it changes the result until proven otherwise.

**Two unit mismatches in the evaluation, both silent.** The shove landed at 0.5 s instead of
5 s, because `eval_set.py` expressed timings as step counts assuming 100 Hz while the trainer
runs at 200 Hz. And every push episode reported "recovered" — one in **-0.005 s** — because
recovery was counted from the moment of the push, while the robot was still upright, so the
hold window was satisfied instantly. Timings are now in seconds with explicit
`step_limit(frequency)` helpers, and recovery only starts counting once the robot has actually
been knocked past the threshold. `disturbed` is now recorded separately from `recovered`.

**Seed variance is large enough that one run is not evidence.** Two runs identical apart from
the seed differed by 12 seconds of total survival (51.3 s vs 39.4 s). Any single-seed headline
is an attack surface.

**The hand-engineered baseline tracks slow speeds worst, not fast ones.** Measured over all
20 episodes, distance travelled against distance commanded:

    0.3 m/s    0.61-0.68 m of 6.0 m     ~11%
    0.6 m/s    4.05-6.25 m of 12.0 m    ~41%
    0.9 m/s    10.65-11.59 m of 18.0 m  ~62%

An initial read of the 0.9-only data said 0.9 m/s was beyond the hardware. The full sweep
says the opposite: tracking improves as the command rises, and at 0.3 m/s Upkie's own
controller moves about 3 cm/s. Lesson: a partial sweep produced a confident and wrong
conclusion about the direction of the effect.

**The grader may be scoring too harshly, and it is our scale, not the controller.** Upkie's
MPC balancer never fell across 20 episodes, survived 5 shoves, recovered in 1.119 s every
time, and never came near a joint limit — and scored **0.1707**. Each measurement looks
correct; the score built from them does not. A headline of the form "our policy beat the
hand-engineered controller" invites the question of why that controller scored 0.17, and
"our weights were harsh" is a weak answer.

**Upkie's own controller saturates its wheel motors.** `peak_torque_frac_wheel` = 1.0, with
the wheels at their 1.7 Nm rating for 4.4% of steps across the full set and 14.9% at 0.9 m/s
alone. Saturation counts as a violation by decision, so the baseline we compare against is
itself in violation. Reported as measured, not special-cased.
