# SPDX-License-Identifier: Apache-2.0
"""The fake trainer. Stage A only. No simulator, no PyBullet, no GPU.

In:  a reward function that has passed Gate 1.
Out: a TrainerResult dict of exactly the same shape the real trainer returns.

Two parts of the output are real and two are made up, and it matters which:

  REAL -- the per-term curves. The reward function is actually called, several
          hundred times, over made-up trajectories. So the term names are the
          model's own names and the term values are what its own arithmetic
          produces. This is what `reflect.py` reads.

  MADE UP -- every physical number in `rollout`. There is no physics here. The
          numbers are generated from a fixed seed so a run is repeatable, and
          they are steered by one quantity, `discrimination`: how much higher
          this reward function scores a trajectory that stays up and hits the
          commanded speed than one that falls over. A reward that cannot tell
          those apart gets bad physical numbers, which is the behaviour the
          pipeline needs in order to be tested.

`trainer_pybullet.py` replaces this file behind the same two functions, `name()`
and `train()`. Nothing else in Koda changes.
"""

import math
import random
import traceback

from koda import SCHEMA_VERSION, eval_set
from koda import state as state_mod

CHECKPOINTS = (0.10, 0.25, 0.50, 1.00)

TRAJECTORY_STEPS = 400


def name() -> str:
    return "stub"


def _mean_total(reward_fn, trajectory):
    """Average summed reward over a trajectory. Raises on a non-finite term."""
    total = 0.0
    for step, test_state in enumerate(trajectory):
        terms = reward_fn(test_state, state_mod.fake_action(test_state))
        for key, value in terms.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"term '{key}' returned {value} at step {step} of training"
                )
            total += value
    return total / len(trajectory)


def _term_curves(reward_fn, discrimination):
    """Per-term values at each checkpoint, by calling the reward function."""
    values = {}
    totals = []
    for fraction in CHECKPOINTS:
        progress = fraction * discrimination
        trajectory = state_mod.fake_trajectory(
            TRAJECTORY_STEPS,
            commanded_speed=0.6,
            wobble=0.30 - 0.22 * progress,
            falls_at=None if progress > 0.35 else int(60 + 300 * progress),
        )
        sums = {}
        for test_state in trajectory:
            terms = reward_fn(test_state, state_mod.fake_action(test_state))
            for key, value in terms.items():
                if not math.isfinite(value):
                    raise ValueError(
                        f"term '{key}' returned {value} at {int(fraction * 100)}% "
                        "of training"
                    )
                sums[key] = sums.get(key, 0.0) + value
        for key, total in sums.items():
            values.setdefault(key, []).append(round(total / len(trajectory), 4))
        totals.append(round(sum(sums.values()) / len(trajectory), 4))

    return {
        "checkpoints": list(CHECKPOINTS),
        "values": values,
        "total": totals,
    }


def _gap(reward_fn, better, worse):
    """How much more this reward likes one trajectory than another, 0 to 1."""
    gap = _mean_total(reward_fn, better) - _mean_total(reward_fn, worse)
    if gap <= 0.0:
        return 0.0
    return gap / (gap + 1.0)


def _discrimination(reward_fn):
    """Two separate questions, because one number cannot stand in for both.

    balance -- does this reward prefer staying up to falling over?
    motion  -- does it prefer driving at the commanded speed to sitting still?

    A reward that pays for being level and penalises movement scores high on
    balance and zero on motion, and the physical numbers it gets have to show
    that. With a single quality number it would look like a good candidate.
    """
    doing_the_job = state_mod.fake_trajectory(TRAJECTORY_STEPS, 0.6, wobble=0.05)
    fallen_over = state_mod.fake_trajectory(TRAJECTORY_STEPS, 0.6, falls_at=40)
    parked = state_mod.fake_trajectory(TRAJECTORY_STEPS, 0.6, wobble=0.05, moves=False)
    return (
        _gap(reward_fn, doing_the_job, fallen_over),
        _gap(reward_fn, doing_the_job, parked),
    )


# The stub has no simulator, so it names its own control rate purely to turn
# the eval set's seconds into a step count.
STUB_FREQUENCY = 100.0


def _rollout(rng, balance, motion):
    """Made-up physical numbers for all 20 evaluation episodes.

    Peak torque deliberately does NOT follow `quality`. It gets its own factor,
    so a candidate can score well on the task and still drive a motor past its
    rated limit -- which is the case `report.html` has to be able to show, and it
    would never occur if every number were a function of one quality value.
    """
    torque_style = rng.uniform(0.75, 1.45)
    episodes = []
    quality = balance
    for episode in eval_set.episodes():
        speed = episode["commanded_speed"]
        jitter = rng.uniform(0.9, 1.1)
        survival = min(1.0, max(0.05, (0.12 + 0.88 * quality) * jitter))
        step_limit = eval_set.step_limit(STUB_FREQUENCY)
        steps = int(step_limit * survival)
        seconds = round(steps / STUB_FREQUENCY, 3)

        record = dict(episode)
        record.update(
            {
                "fell": steps < step_limit,
                "episode_seconds": seconds,
                "steps_survived": steps,
                "step_limit": step_limit,
                "mean_speed_error": round(speed * 0.95 * (1.0 - motion) * jitter, 4),
                "net_displacement_m": round(
                    speed * seconds * (0.05 + 0.95 * motion) * jitter, 4
                ),
                "heading_error_deg": round(28.0 * (1.0 - quality) * jitter, 2),
                "max_pitch_excursion_deg": round(
                    (4.0 + 46.0 * (1.0 - quality)) * jitter, 2
                ),
                "closest_joint_margin_deg": round((1.5 + 9.0 * quality) * jitter, 2),
                "peak_torque_nm": {
                    "leg": round(
                        (4.0 + 9.0 * (1.0 - quality)) * jitter * torque_style, 3
                    ),
                    "wheel": round(
                        (0.95 + 0.55 * (1.0 - quality)) * jitter * torque_style, 3
                    ),
                },
                "rms_torque_nm": {
                    "leg": round(
                        (1.6 + 3.0 * (1.0 - quality)) * jitter * torque_style, 3
                    ),
                    "wheel": round(
                        (0.35 + 0.25 * (1.0 - quality)) * jitter * torque_style, 3
                    ),
                },
            }
        )

        if episode["kind"] == "push":
            recovered = rng.random() < quality
            record["recovered"] = recovered
            record["recovery_time_s"] = (
                round(0.4 + 2.6 * (1.0 - quality) * jitter, 3)
                if recovered
                else seconds  # the episode cap: never recovered
            )
        episodes.append(record)

    return {"schema_version": SCHEMA_VERSION, "episodes": episodes}


def train(
    reward_fn,
    candidate_id: str,
    round_number: int,
    goal: str,
    seed: int = 0,
    total_steps: int | None = None,
    out_dir: str = "runs",
    eval_episodes: int | None = None,
) -> dict:
    """Pretend to train one candidate. Returns a TrainerResult dict.

    Takes the same arguments as `trainer_pybullet.train` and ignores the ones
    that only mean something for real training. The two signatures have to
    match: the orchestrator picks a trainer by name and calls it the same way,
    so a divergence here is only discovered at run time.
    """
    rng = random.Random(f"{candidate_id}:{round_number}:{seed}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "round": round_number,
        "trainer": name(),
        "status": "trained",
        "error": None,
        "terms": None,
        "rollout": None,
        "policy_path": None,
        "video_path": None,
        "train_seconds": 0.0,
        "compute_seconds": 0.0,
    }

    try:
        balance, motion = _discrimination(reward_fn)
        terms = _term_curves(reward_fn, balance)
    except Exception:
        result["status"] = "dead_nan"
        result["error"] = traceback.format_exc()
        return result

    # A real training run sometimes diverges for reasons Gate 1 cannot see.
    # One candidate in roughly twelve, deterministically, so the orchestrator's
    # dead-in-training path gets exercised on every run.
    if rng.random() < 0.08:
        result["status"] = "dead_gate2"
        result["error"] = (
            "RuntimeError: policy loss became non-finite at step 18,432 of "
            "200,000 (this is the stub trainer's simulated divergence)"
        )
        return result

    # Rounds later in the search start from the previous winner's design, so
    # they land a little better on average.
    round_bonus = 0.06 * (round_number - 1)
    noise = rng.uniform(-0.05, 0.05)
    balance = min(0.98, max(0.02, balance + round_bonus + noise))
    motion = min(0.98, max(0.02, motion + round_bonus + noise))

    result["terms"] = terms
    result["rollout"] = _rollout(rng, balance, motion)
    result["train_seconds"] = round(rng.uniform(280.0, 620.0), 1)
    result["compute_seconds"] = result["train_seconds"]
    return result
