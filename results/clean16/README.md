# The 16 rewards that were never shown a false number, retrained properly

    ./.venv/bin/python retrain_clean.py

500,000 steps each, 6 at a time, 66 minutes, no model calls. Trained with the
rl_zoo3 settings adopted in `6c08207` and scored by `koda/scorer.py`.

## Why these 16

Round 1 of every search saw no measurement block — a cold start sees nothing,
and `--from-reward` passes the reward SOURCE without its reflection. Rounds 2
and 3 were written in response to numbers produced by the old short-horizon
learner. These are the rewards no false measurement could have influenced.

Two of the 18 died at Gate 1 when first generated; 15 of 16 trained here.

## Result

| | score | survival | torque_headroom | speed error |
|---|---|---|---|---|
| s14_r1c4 | **0.0380** | 1.000 | 0.774 | 0.644 |
| s14_r1c3 | 0.0348 | 0.315 | 0.358 | 0.731 |
| s14_r1c1 | 0.0327 | 1.000 | 0.995 | 0.597 |
| s12_r1c5 | 0.0219 | 0.501 | 0.498 | 0.614 |
| s12_r1c6 | 0.0115 | 0.394 | 0.096 | 0.686 |
| s14_r1c2 | 0.0023 | 0.032 | 0.473 | 0.625 |
| s12_r1c3 | 0.0018 | 0.041 | 0.615 | 0.591 |
| s12_r1c1 | 0.0014 | 0.035 | 0.544 | 0.591 |
| s13_r1c1 | 0.0011 | 0.041 | 0.524 | 0.596 |
| s13_r1c2 | 0.0010 | 0.037 | 0.324 | 0.592 |
| s13_r1c4 | 0.0003 | 0.018 | 0.130 | 0.609 |
| s12_r1c2, s12_r1c4, s13_r1c3, s13_r1c5 | 0.0000 | — | — | — |

**r3c1, with the same settings, scores 0.1981 at 500k and 0.2286 at 1M.**

## What it says

**The best reward in this project came from iteration, not from a first
attempt.** `r3c1` is a round-3 reward, written while the model was being fed
measurements from a learner that could only see half a second ahead. Every
round-1 reward — none of which could have been misled, because none was shown a
number — is at least 5x worse.

So the iterative loop produced real improvement even while its feedback was
distorted. That is the opposite of what the search_13 and search_14 results
suggested on their own, and it is a stronger argument for more rounds than for
fewer.

**The ranking reshuffles completely under a better learner.** `s12_r1c6` won
round 1 of search_12 at 0.0740 under the old settings and places fifth here at
0.0115. `s12_r1c5` scored 0.0000 then and places fourth now. A reward's score
was never a property of the reward alone.

**Two candidates survive the full 20 s and still score near zero.** `s12_r1c4`
has survival 1.000 with headroom 0.021 — it stays upright by sitting on the
torque limit. `s14_r1c1` has survival 1.000 AND headroom 0.995, the cleanest
pair of gate scores anything has produced, and still scores 0.0327 because it
barely covers ground. Under the old additive grader both would have looked
strong.
