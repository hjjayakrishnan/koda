# SPDX-License-Identifier: Apache-2.0
"""The grader. This is the real one.

**Nemotron never sees this file.** It sees grade values, through `reflect.py`,
and never grading code.

In:  `TrainerResult["rollout"]` -- physical measurements and nothing else. No
     reward values, no reward term names, no fitness signal. That is enforced by
     the signature: `rollout` is the only data argument there is.
Out: a `ScoreResult` as specified in INTERFACE.md section 4.

### What this file is allowed to measure, and what it must not

| Kind of reward term | May the grader measure it? |
|---|---|
| Regularisation -- a penalty on *how* the motion is made: action rate, jerk, joint acceleration, vertical velocity, energy magnitude | **Never.** Mirroring these puts a human's answer inside the grader. Three rounds of feedback would then steer the model toward our own term list, and the comparison collapses into "did it rediscover our vocabulary" |
| Task achievement -- *was the commanded goal met* | **Must.** A grader that does not measure the task measures nothing |
| Hardware safety -- *would this break the robot* | **Must**, and only as a threshold that is crossed or not. Never as a continuous penalty |

The deliberate blind spots are action rate, jerk, vibration and energy
magnitude -- anything about motion smoothness. They are preserved on purpose.
A pathology that survives this grader is the finding, not a bug to patch.

**No reward term name appears anywhere in this file.** If one ever does, the
boundary has broken.
"""

import math
import hashlib
from pathlib import Path

from koda import SCHEMA_VERSION
from koda import state as state_mod

# --------------------------------------------------------------------------
# What the score is computed over. Pre-registered, frozen at `prereg-v1`.
# --------------------------------------------------------------------------

ACTIVE_MEASUREMENTS = (
    "survival_frac",
    "mean_speed_error",
    "displacement_frac",
    "heading_error_deg",
)

# Staying upright is a GATE, not a share of the score.
#
# It was a 0.45 additive share until R0 caught the problem: a robot that stands
# perfectly still and ignores the command banked 45% of the maximum for doing
# nothing, and the spin-in-place cheat scored 0.432. Surviving is not partial
# credit for the task -- it is the precondition for having performed it at all.
#
# So: score = survival x (how well the commanded task was done) x heading.
# A policy that never moves now scores near zero however long it stays up, and
# one that falls immediately cannot score at all.
WEIGHTS = {
    "speed": 0.55,
    "displacement": 0.45,
}

# Velocity tracking uses the standard exponential form from `legged_gym` and
# Isaac Lab (`track_lin_vel_xy_exp`), which is what essentially all legged
# locomotion RL uses:
#
#     tracking = exp(-error**2 / tracking_sigma)        tracking_sigma = 0.25
#
# The constant is named SPEED_SIGMA rather than tracking_sigma purely because
# the grader is forbidden from using reward-term vocabulary in its own
# identifiers, and tests/test_scorer.py enforces that against the source. It
# is the same quantity as legged_gym's tracking_sigma.
#
# We did not invent a curve. Our contribution is automated reward design, not
# grader design, so the grader uses the field's standard metric and can be
# checked against any locomotion paper.
#
# It replaced a linear `max(0, 1 - error/0.3)` on 2026-10-27, before prereg-v1
# and before any search was run. That version had a dead zone: any mean error
# at or above 0.3 m/s scored exactly 0, so 0.32 m/s and 2.0 m/s were
# indistinguishable across 55% of the score. Upkie's own MPC balancer sits at
# 0.321 m/s, so the most heavily weighted term was pinned at zero for the
# baseline and would have been for most candidates. Since the grader also picks
# each round's winner, a flat term means selection ignores it entirely.
#
#     mean error   linear (old)   exp (now)
#       0.20          0.333         0.852
#       0.321         0.000         0.662     <- Upkie MPC baseline
#       0.50          0.000         0.368
#       1.00          0.000         0.018
#
# No dead zone, gradient everywhere a candidate might land.
SPEED_SIGMA = 0.25
HEADING_SCALE_DEG = 45.0  # degrees of worst drift that takes the score to zero

# --------------------------------------------------------------------------
# Hardware thresholds. Read from the installed upkie model, not guessed.
# --------------------------------------------------------------------------
#
# **Reaching 100% of an actuator's rated torque is a violation.** User decision,
# 2026-10-27. Per actuator class: wheels 1.7 Nm, legs 16.0 Nm.
#
# This flags Upkie's own shipped controllers, which were measured at exactly
# 1.7000 Nm -- 100% of the mj5208 rating -- just to stand still. That is
# reported exactly as measured. The hand-engineered baseline is NOT exempted and
# the number is not rounded down: a shipped controller maxing out the motors it
# runs on is a real observation about this robot, and hiding it to tidy up the
# comparison would be the dishonest move.
#
# TOLERANCE settles the floating-point question. The simulator clamps a
# commanded torque at the rated value, so a clipped reading is exactly 1.7 while
# an arithmetically-identical one can land at 1.6999999 through rounding. The
# test is therefore `>= 1.0 - TOLERANCE`, so both are treated the same way.
# Tolerance is written down rather than left to a tie.

TORQUE_TOLERANCE = 0.001  # within 0.1% of rated counts as having reached it

VIOLATIONS = (
    "wheel_torque_exceeded",
    "leg_torque_exceeded",
    "joint_limit_hit",
    "fell_over",
    "never_recovered_from_push",
)


def name() -> str:
    return "koda-v1"


def scorer_sha() -> str:
    """Sha of this file **as actually run**.

    Not pinned to the pre-registered value -- pinning would stop it detecting
    the mismatch that makes it useful. A sha differing from the one recorded at
    `prereg-v1` *is* the disclosure that the grader changed.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7]


# --------------------------------------------------------------------------
# Reducers. Three of these are not means, and that is deliberate.
# --------------------------------------------------------------------------


def _survival(episodes):
    return sum(e["steps_survived"] / e["step_limit"] for e in episodes) / len(
        episodes
    )


def _displacement_frac(episode):
    """Ground covered as a fraction of what the command implies.

    Returns None when the robot was asked to hold still. The quantity is
    undefined for a zero command -- dividing by a guarded epsilon produced
    -93300.0 on the PD baseline, which is not a small number, it is a
    meaningless one. A stationary episode is excluded from this measure rather
    than scored on it.

    Negative is kept, not clamped: travelling backwards is informative.
    """
    wanted = episode["commanded_speed"] * episode["episode_seconds"]
    if abs(wanted) < 1e-6:
        return None
    return episode["net_displacement_m"] / wanted


def _reached_limit(fraction):
    """True when a torque reading has reached its rated limit.

    `>= 1.0 - TORQUE_TOLERANCE`, so a value clipped to exactly the rating and
    one that lands a rounding error below it are treated identically.
    """
    return fraction >= 1.0 - TORQUE_TOLERANCE


def _saturated_fractions(episodes):
    """Share of control steps spent at the rated limit, per actuator class.

    Needs `saturated_steps` in the rollout, which `trainer_pybullet.py` counts
    while it runs. Returns None for a rollout that predates that field rather
    than inventing a number for it.
    """
    out = {}
    for kind in ("leg", "wheel"):
        counted = 0
        total = 0
        for episode in episodes:
            saturated = episode.get("saturated_steps")
            if not saturated or kind not in saturated:
                return {"leg": None, "wheel": None}
            counted += saturated[kind]
            total += episode["steps_survived"]
        out[kind] = round(counted / total, 4) if total else None
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _task_score(episodes):
    """Score one group of episodes, 0.000 to 1.000."""
    if not episodes:
        return 0.0

    survival = _survival(episodes)
    speed_error = sum(e["mean_speed_error"] for e in episodes) / len(episodes)
    displacement = _mean([_displacement_frac(e) for e in episodes])
    worst_heading = max(e["heading_error_deg"] for e in episodes)

    speed_part = math.exp(-(speed_error * speed_error) / SPEED_SIGMA)
    heading_factor = max(0.0, 1.0 - worst_heading / HEADING_SCALE_DEG)

    if displacement is None:
        # Every episode here was stationary, so displacement is undefined and
        # the task is entirely "match the commanded speed", which was zero.
        task = speed_part
    else:
        displacement_part = min(1.0, max(0.0, displacement))
        task = (
            WEIGHTS["speed"] * speed_part
            + WEIGHTS["displacement"] * displacement_part
        )

    return round(survival * task * heading_factor, 4)


def score(rollout: dict, candidate_id: str = "candidate") -> dict:
    """Grade one candidate from its physical measurements alone."""
    episodes = rollout["episodes"]
    if not episodes:
        raise ValueError("cannot score an empty rollout")

    task = [e for e in episodes if e["kind"] == "task"]
    push = [e for e in episodes if e["kind"] == "push"]
    if not task:
        raise ValueError("a rollout needs at least one task episode")

    peak_leg = max(e["peak_torque_nm"]["leg"] for e in episodes)
    peak_wheel = max(e["peak_torque_nm"]["wheel"] for e in episodes)
    leg_frac = peak_leg / state_mod.LEG_TORQUE_LIMIT_NM
    wheel_frac = peak_wheel / state_mod.WHEEL_TORQUE_LIMIT_NM

    recovered = [e for e in push if e.get("recovered")]
    fell = [e for e in episodes if e.get("fell")]

    # Two numbers, not one. `peak` is the highest torque seen and is what the
    # violation fires on. `saturated_step_frac` is the share of control steps
    # sitting at the limit, and it is the one that carries meaning: touching the
    # ceiling once while recovering from a shove is ordinary engineering,
    # sitting at 100% for twenty seconds destroys the motor. The violation flag
    # alone cannot tell those apart, and once the baseline violates too it stops
    # separating good from bad by itself.
    saturated = _saturated_fractions(episodes)

    measurements = {
        "survival_frac": round(_survival(task), 4),
        "mean_speed_error": round(
            sum(e["mean_speed_error"] for e in task) / len(task), 4
        ),
        "displacement_frac": (
            round(v, 4)
            if (v := _mean([_displacement_frac(e) for e in task])) is not None
            else None
        ),
        "heading_error_deg": round(max(e["heading_error_deg"] for e in task), 2),
        "max_pitch_excursion_deg": round(
            max(e["max_pitch_excursion_deg"] for e in episodes), 2
        ),
        "closest_joint_margin_deg": round(
            min(e["closest_joint_margin_deg"] for e in episodes), 2
        ),
        "peak_torque_frac_leg": round(leg_frac, 4),
        "peak_torque_frac_wheel": round(wheel_frac, 4),
        "saturated_step_frac_leg": saturated["leg"],
        "saturated_step_frac_wheel": saturated["wheel"],
        # Averaged over recovered episodes only. The count that never recovered
        # is reported beside it, never averaged in as a slow recovery.
        "recovery_time_s": (
            round(sum(e["recovery_time_s"] for e in recovered) / len(recovered), 3)
            if recovered
            else None
        ),
        "non_recovery_count": len(push) - len(recovered),
        "fell_count": len(fell),
        "episode_count": len(episodes),
    }

    violations = []
    if _reached_limit(wheel_frac):
        violations.append("wheel_torque_exceeded")
    if _reached_limit(leg_frac):
        violations.append("leg_torque_exceeded")
    if measurements["closest_joint_margin_deg"] <= 0.0:
        violations.append("joint_limit_hit")
    if fell:
        violations.append("fell_over")
    if push and not recovered:
        violations.append("never_recovered_from_push")

    per_speed = {}
    for speed in sorted({e["commanded_speed"] for e in task}):
        per_speed[str(speed)] = _task_score(
            [e for e in task if e["commanded_speed"] == speed]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "scorer": name(),
        "score": _task_score(task),
        "per_speed": per_speed,
        "measurements": measurements,
        "active_measurements": list(ACTIVE_MEASUREMENTS),
        "violations": violations,
        "scorer_sha": scorer_sha(),
    }
