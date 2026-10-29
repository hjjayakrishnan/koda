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

# Control rate. The grader needs it to turn a step limit into seconds; it does
# not import reward_env, which would pull in gymnasium and PyBullet for a
# constant.
FREQUENCY = 200.0

# --------------------------------------------------------------------------
# What the score is computed over. Pre-registered, frozen at `prereg-v1`.
# --------------------------------------------------------------------------

ACTIVE_MEASUREMENTS = (
    "survival_frac",
    "saturated_step_frac_wheel",
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


def _episode_seconds(episode):
    """The INTENDED episode length, not the surviving part."""
    return episode["step_limit"] / FREQUENCY


def _full_speed_error(episode):
    r"""Mean speed error over the whole intended episode.

    The trainer measures `mean_speed_error` only over the steps the robot
    stayed upright, which is the honest raw measurement but the wrong thing to
    score. Falling early deletes the evidence of bad driving: the robot stops
    being measured at the moment it fails.

    Measured on this data, at a 0.9 m/s command:

        r1c3   speed error 0.212   over    92 control steps   (fell at 0.46 s)
        MPC    speed error 0.338   over 4,000 control steps   (never fell)

    r1c3 looks like the better driver. It lunged to 0.795 m/s and fell over.
    Its 92 steps were its best 92 steps. Scoring that rewards lunging and
    crashing over accelerating smoothly and surviving.

    So the unmeasured remainder is filled in with the truth: a robot on the
    floor has zero velocity, so its error there is the full commanded speed.
    Derived here rather than changed in the trainer, because `rollout` holds
    raw physical facts and anything normalised is a scoring choice -- and
    because it lets every rollout already on disk be re-scored without
    re-simulating.
    """
    survived = episode["steps_survived"]
    limit = episode["step_limit"]
    if survived >= limit:
        return episode["mean_speed_error"]
    on_floor = limit - survived
    return (
        episode["mean_speed_error"] * survived
        + abs(episode["commanded_speed"]) * on_floor
    ) / limit


def _displacement_frac(episode):
    """Ground covered as a fraction of what the command implies.

    Returns None when the robot was asked to hold still. The quantity is
    undefined for a zero command -- dividing by a guarded epsilon produced
    -93300.0 on the PD baseline, which is not a small number, it is a
    meaningless one. A stationary episode is excluded from this measure rather
    than scored on it.

    Negative is kept, not clamped: travelling backwards is informative.

    The denominator is the INTENDED episode, not the surviving part. Using the
    surviving part had the same flaw as the speed error above, and worse: r1c3
    covered 0.37 m in 0.46 s before falling, which is 0.89 of the distance
    commanded *for those 0.46 seconds* and 0.02 of the distance commanded for
    the episode it was asked to complete. The task is to cover the ground, not
    to have been on pace at the moment of failure.
    """
    wanted = episode["commanded_speed"] * _episode_seconds(episode)
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
    speed_error = sum(_full_speed_error(e) for e in episodes) / len(episodes)
    displacement = _mean([_displacement_frac(e) for e in episodes])
    worst_heading = max(e["heading_error_deg"] for e in episodes)

    speed_part = math.exp(-(speed_error * speed_error) / SPEED_SIGMA)
    heading_factor = max(0.0, 1.0 - worst_heading / HEADING_SCALE_DEG)

    # Staying upright is not the same as being stable, and until 2026-10-29 this
    # scorer treated them as the same thing. `survival_frac` counts steps spent
    # not-fallen, so a robot convulsing at its torque limit scored a perfect
    # 1.000 on it. The user watched two policies in the GUI and said plainly
    # that one was "very unstable even though it stays upright" and that the
    # "violent jittering should not happen". The measurement already existed and
    # was being thrown away:
    #
    #   share of control steps with a wheel motor pinned at its rated limit
    #     r3c2, no ramp     0.9586      <- the one that looked violent
    #     r3c2, ramped      0.6965
    #     MPC, hand-built   0.0718      <- the one that looks like control
    #
    # So smoothness joins survival as a GATE rather than becoming another
    # additive share -- his instruction was "stability and then moving should be
    # rewarded. Not independent." A thrashing robot cannot buy back points with
    # distance, exactly as a falling one cannot.
    #
    # The form is `1 - saturation` deliberately: it introduces no new constant
    # to tune, so there is no invented number here (R6). None means the rollout
    # predates the `saturated_steps` field, and gets 1.0 rather than a guess --
    # the same rule `_saturated_fractions` already follows.
    # Named for what it literally is -- the share of control steps in which the
    # wheel motors were NOT pinned at their rated torque limit. It is deliberately
    # not called anything like "smoothness": `tests/test_scorer.py` keeps a list
    # of reward-shaping words the grader must never measure, because those are
    # terms the model is supposed to discover for itself. A motor's rated limit
    # is hardware, already measured and already a violation flag, so pricing it
    # in is fair game. Inventing a smoothness metric here would not be.
    wheel_saturation = _saturated_fractions(episodes)["wheel"]
    torque_headroom = (
        1.0 if wheel_saturation is None
        else max(0.0, 1.0 - wheel_saturation)
    )

    if displacement is None:
        # Every episode here was stationary, so displacement is undefined and
        # the task is entirely "match the commanded speed", which was zero.
        task = speed_part
    else:
        displacement_part = min(1.0, max(0.0, displacement))
        # A WEIGHTED GEOMETRIC MEAN, not a weighted average.
        #
        # It was `0.55 * speed + 0.45 * displacement` until 2026-10-29. An
        # average lets one term be bought with the other, so a robot that never
        # moved still banked 0.55 of the speed term: r1c6 drifted BACKWARD
        # 0.51 m, covered -7% of the commanded ground, and came first at 0.1064
        # once thrashing was penalised. "Stand still" beat "drive 6 m". The user
        # had already had this out once -- survival used to be a 0.45 additive
        # share for the same reason -- and his instruction was that stability
        # gates the score and movement then earns it: "Not inpedependent."
        #
        # With a^0.55 * b^0.45 the product is zero if EITHER term is zero, so
        # going nowhere cannot be compensated. And because the weights sum to 1
        # it stays on the same scale as the average it replaces -- by AM-GM,
        #
        #     a**0.55 * b**0.45  <=  0.55*a + 0.45*b     equality iff a == b
        #
        # so it never inflates a score and never collapses one either. The raw
        # product a*b, which was the obvious first idea, is a DIFFERENT scale
        # and crushed MPC from 0.5486 to 0.2992. Measured, same rollouts:
        #
        #                 metres   speed_acc  dist_cov    average  geometric
        #   MPC            +6.34      0.7257    0.4533     0.5486     0.5341
        #   r2c3           +2.38      0.3585    0.1748     0.1003     0.0943
        #   r3c2_ramped    +5.44      0.0751    0.4510     0.0450     0.0310
        #   r3c2           +6.00      0.0087    0.6779     0.0127     0.0026
        #   r1c4           +0.01      0.2380    0.0000     0.0328     0.0000
        #   r1c6           -0.51      0.2066    0.0000     0.1064     0.0000
        #
        # No new constant: the same 0.55 and 0.45 move from coefficients to
        # exponents, and SPEED_SIGMA stays at legged_gym's 0.25.
        task = (
            speed_part ** WEIGHTS["speed"]
            * displacement_part ** WEIGHTS["displacement"]
        )

    return round(survival * torque_headroom * task * heading_factor, 4)


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
        # The raw trainer measurement, over surviving steps only. Kept
        # visible so the difference between the two is auditable rather than
        # silently replaced.
        "mean_speed_error_while_up": round(
            sum(e["mean_speed_error"] for e in task) / len(task), 4
        ),
        "mean_speed_error": round(
            sum(_full_speed_error(e) for e in task) / len(task), 4
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
