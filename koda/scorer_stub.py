# SPDX-License-Identifier: Apache-2.0
"""A stand-in grader, so Stage A can run end to end.

THIS IS NOT THE PROJECT'S GRADER. `scorer.py` is written and owned by the other
pane, is red-teamed against three hand-written cheating rewards, and is frozen at
`git tag prereg-v1`. This file exists only so the orchestrator has something to
call before that one lands, and it must not be used for any scored run.

The one thing it does take seriously is the leakage boundary. It is handed
`TrainerResult["rollout"]` and nothing else -- physical measurements only, no
reward values, no term names, no fitness signal. Swapping the real grader in is
one import line in `orchestrator.py`.

In:  a rollout dict (20 episodes of physical measurements).
Out: a ScoreResult dict.
"""

import hashlib
from pathlib import Path

from koda import SCHEMA_VERSION
from koda import state as state_mod

# Scores are computed over these measurements. Pre-registered, because scores
# from searches with different lists are not comparable.
ACTIVE_MEASUREMENTS = (
    "survival_frac",
    "mean_speed_error",
    "displacement_frac",
    "heading_error_deg",
)

SPEED_ERROR_SCALE = 0.3  # m/s of mean error that takes the speed part to zero
HEADING_SCALE_DEG = 45.0  # degrees of worst drift that takes the score to zero


def name() -> str:
    return "stub"


def scorer_sha() -> str:
    """Sha of this file as actually run. The disclosure if the grader changed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7]


def _task_score(episodes):
    if not episodes:
        return 0.0
    survival = sum(e["steps_survived"] / e["step_limit"] for e in episodes) / len(episodes)
    speed_error = sum(e["mean_speed_error"] for e in episodes) / len(episodes)
    displacement = sum(
        e["net_displacement_m"] / max(1e-6, e["commanded_speed"] * e["episode_seconds"])
        for e in episodes
    ) / len(episodes)
    worst_heading = max(e["heading_error_deg"] for e in episodes)

    speed_part = max(0.0, 1.0 - speed_error / SPEED_ERROR_SCALE)
    displacement_part = min(1.0, max(0.0, displacement))
    heading_factor = max(0.0, 1.0 - worst_heading / HEADING_SCALE_DEG)

    base = 0.45 * survival + 0.30 * speed_part + 0.25 * displacement_part
    return round(base * heading_factor, 4)


def score(rollout: dict, candidate_id: str = "candidate") -> dict:
    """Grade one candidate from its physical measurements alone."""
    episodes = rollout["episodes"]
    task = [e for e in episodes if e["kind"] == "task"]
    push = [e for e in episodes if e["kind"] == "push"]

    survival = sum(e["steps_survived"] / e["step_limit"] for e in task) / len(task)
    speed_error = sum(e["mean_speed_error"] for e in task) / len(task)
    displacement = sum(
        e["net_displacement_m"] / max(1e-6, e["commanded_speed"] * e["episode_seconds"])
        for e in task
    ) / len(task)

    peak_leg = max(e["peak_torque_nm"]["leg"] for e in episodes)
    peak_wheel = max(e["peak_torque_nm"]["wheel"] for e in episodes)

    recovered = [e for e in push if e.get("recovered")]
    measurements = {
        "survival_frac": round(survival, 4),
        "mean_speed_error": round(speed_error, 4),
        "displacement_frac": round(displacement, 4),
        "heading_error_deg": round(max(e["heading_error_deg"] for e in task), 2),
        "max_pitch_excursion_deg": round(
            max(e["max_pitch_excursion_deg"] for e in episodes), 2
        ),
        "closest_joint_margin_deg": round(
            min(e["closest_joint_margin_deg"] for e in episodes), 2
        ),
        "peak_torque_frac_leg": round(peak_leg / state_mod.LEG_TORQUE_LIMIT_NM, 4),
        "peak_torque_frac_wheel": round(peak_wheel / state_mod.WHEEL_TORQUE_LIMIT_NM, 4),
        # Averaged only over episodes where the robot actually recovered. The
        # count of the ones where it never did is reported next to it, never
        # averaged in as though it were a slow recovery.
        "recovery_time_s": (
            round(sum(e["recovery_time_s"] for e in recovered) / len(recovered), 3)
            if recovered
            else None
        ),
        "non_recovery_count": len(push) - len(recovered),
    }

    violations = []
    if peak_wheel > state_mod.WHEEL_TORQUE_LIMIT_NM:
        violations.append("wheel_torque_exceeded")
    if peak_leg > state_mod.LEG_TORQUE_LIMIT_NM:
        violations.append("leg_torque_exceeded")
    if measurements["closest_joint_margin_deg"] <= 0.0:
        violations.append("joint_limit_hit")
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
