# What the model actually receives between rounds

This is the real text sent to Nemotron 3 Super at the start of round 3 of
`search_15`, copied out of `results/search_15/run.json` verbatim. It is produced
by `koda/reflect.py` from the winning candidate's measurements.

The model never sees `koda/scorer.py`. It sees what the robot did and the score
that produced, and has to connect them itself.

---

```
Each reward term, averaged per control step, at 10%, 25%, 50%, 100% of training:
    heartbeat         -0.000 -> -0.000 -> -0.000 -> +0.000
    jerk_penalty      -0.000 -> -0.000 -> -0.000 -> -0.000
    speed_ratio       -0.257 -> -0.156 -> -0.167 -> -0.185
    stability_gauss   -0.192 -> -0.118 -> +0.020 -> +0.248
    yaw_alignment_l1  +0.000 -> +0.000 -> +0.000 -> +0.000
    (sum)             -0.449 -> -0.275 -> -0.147 -> +0.063

The term 'yaw_alignment_l1' read +0.000 at every checkpoint. It never varied with what the robot did.
The term 'heartbeat' read -0.000 at every checkpoint. It never varied with what the robot did.

Measured on the trained robot, over 20 evaluation episodes:
    stayed up for 60% of the episode, on average
    mean speed error 0.602 m/s, against commands of 0.3, 0.6 and 0.9 m/s
    covered 1% of the ground the command implies
    worst heading drift 0.0 degrees
    worst pitch excursion 58.7 degrees
    closest a joint came to its travel limit: 70.6 degrees
    nothing ever knocked it far enough off upright to count as a disturbance
    peak motor torque reached 100% of the wheel motors' rated limit and 30% of the leg motors'
    wheel motors sat pinned at that limit for 1% of every step taken
    limits exceeded: wheel_torque_exceeded, fell_over

Score: 0.036 out of 1.000 (0.3 m/s: 0.024, 0.6 m/s: 0.000, 0.9 m/s: 0.015)
```

---

## What is in it, and what is deliberately not

**In it:** every reward term the candidate defined, averaged per control step at
10%, 25%, 50% and 100% of training — so the model can see which of its own terms
actually moved. Then the physical measurements: how long the robot stayed up,
how far off the commanded speed it was, how much ground it covered, how hard it
leaned, how close a joint came to its travel limit, how much of every step the
wheel motors spent pinned at their torque limit, and which hardware limits were
exceeded. Finally the score, broken out per commanded speed.

**Not in it:** the scoring formula. The model is never told that the score is
`stayed_upright x torque_headroom x speed_accuracy^0.55 x distance_covered^0.45
x straightness`. It is given the measurements and the number they produced.

**Also not in it:** any reward-shaping term we might have handed over.
`tests/test_scorer.py` keeps a list — `jerk`, `action_rate`, `energy`,
`smoothness`, `vibration` — and asserts the grader's code measures none of them,
so the model has to invent those itself. Torque saturation is reported because
it is a hardware limit that was already being measured, not a shaping choice we
made for the model.

## The line that was added on 2026-10-29

```
wheel motors sat pinned at that limit for N% of every step taken
```

Before this, the feedback reported only *peak* torque, which cannot tell apart a
policy that touches the limit once while recovering from a shove and one that
sits on it for twenty seconds. The best candidate of the first search sat at the
limit on **96%** of its control steps; the hand-written MPC controller does it on
**7%**. Both reported the same peak.
