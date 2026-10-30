# r3c1 retrained at 1,000,000 steps — a negative result

    ./.venv/bin/python train_one.py \
        --reward results/search_12/candidates/round3/r3c1.py \
        --steps 1000000 --seed 11 --id r3c1_1M

Same reward function, same seed, no setpoint ramp. The only change is twice the
training. 36 minutes, no cloud, nothing spent.

## The question it was asked

`r3c1` falls in 36% of its episodes. Is that the reward function's fault, or was
500,000 steps simply not enough training?

## The answer: not training

| | score | survival | torque_headroom | speed error | distance |
|---|---|---|---|---|---|
| r3c1 @ 500k | 0.1468 | 0.638 | 0.914 | 0.508 | +0.162 |
| r3c1 @ 1M | 0.1617 | **0.622** | 0.881 | **0.508** | +0.234 |
| MPC, hand-built | 0.5341 | 1.000 | 0.928 | 0.283 | +0.453 |

Survival went **down**, 0.638 to 0.622. Speed error is identical to three
decimal places. Doubling the training bought nothing in either of the two things
that matter, so 500,000 steps was already enough and the reward function is the
ceiling.

The score did rise, 0.1468 to 0.1617, entirely from distance covered: +0.162 to
+0.234. It drives harder and slightly less carefully — `torque_headroom` slipped
from 0.914 to 0.881. That is a 10% score gain from recklessness, not control.

## One number not to misread

The training log reports per-episode speed errors of 0.261 to 0.432, which looks
much better than the 0.508 headline. That is `mean_speed_error_while_up`,
measured only over the steps the robot stayed upright. The scored figure fills
the time on the floor with the full commanded speed as error, because a robot
lying down is not tracking anything. Falling early erases the gain. Both numbers
are kept in every result file so the difference is always auditable.

## What it decided

Training length is ruled out. More search is the justified spend, which is what
`--from-reward` and the best-so-far parent fix were built for.
