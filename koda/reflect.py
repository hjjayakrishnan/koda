# SPDX-License-Identifier: Apache-2.0
"""Turn one candidate's training logs and grades into the text the model reads.

In:  a TrainerResult and its ScoreResult.
Out: plain English, which `prompt.py` pastes into the next round.

This is the only place in Koda where grade *values* reach the model. Grade
*code* never does: this function is handed the grader's output, not the grader.

It reports facts and stops. It does not suggest reward terms, name terms the
model should add, or hint at what the grader is blind to. The design decisions
stay the model's, which is the only reason the result means anything.
"""


def _curve(values):
    return " -> ".join(f"{v:+.3f}" for v in values)


def _flat_terms(terms):
    """Terms whose value never changed across the whole run."""
    flat = []
    for key, values in terms["values"].items():
        if max(values) - min(values) < 1e-9:
            flat.append((key, values[0]))
    return flat


def reflect(trainer_result: dict, score_result: dict | None) -> str:
    """The written report for one candidate."""
    lines = []

    if trainer_result["status"] != "trained":
        lines.append(
            f"This candidate did not finish: {trainer_result['status']}. "
            "The exact error is given above."
        )
        return "\n".join(lines)

    terms = trainer_result["terms"]
    checkpoints = ", ".join(f"{int(c * 100)}%" for c in terms["checkpoints"])
    lines.append(
        f"Each reward term, averaged per control step, at {checkpoints} of training:"
    )
    width = max(len(k) for k in terms["values"])
    for key, values in sorted(terms["values"].items()):
        lines.append(f"    {key:<{width}}  {_curve(values)}")
    lines.append(f"    {'(sum)':<{width}}  {_curve(terms['total'])}")

    flat = _flat_terms(terms)
    if flat:
        lines.append("")
        for key, value in flat:
            lines.append(
                f"The term '{key}' read {value:+.3f} at every checkpoint. It never "
                "varied with what the robot did."
            )

    if score_result is None:
        return "\n".join(lines)

    m = score_result["measurements"]
    lines.append("")
    lines.append("Measured on the trained robot, over 20 evaluation episodes:")
    lines.append(
        f"    stayed up for {m['survival_frac'] * 100:.0f}% of the episode, on average"
    )
    lines.append(
        f"    mean speed error {m['mean_speed_error']:.3f} m/s, against commands "
        "of 0.3, 0.6 and 0.9 m/s"
    )
    lines.append(
        f"    covered {m['displacement_frac'] * 100:.0f}% of the ground the command "
        "implies"
    )
    lines.append(f"    worst heading drift {m['heading_error_deg']:.1f} degrees")
    lines.append(
        f"    worst pitch excursion {m['max_pitch_excursion_deg']:.1f} degrees"
    )
    lines.append(
        f"    closest a joint came to its travel limit: "
        f"{m['closest_joint_margin_deg']:.1f} degrees"
    )

    if m["recovery_time_s"] is None:
        lines.append("    never recovered from any of the 5 shoves")
    else:
        recovered = 5 - m["non_recovery_count"]
        lines.append(
            f"    recovered from {recovered} of 5 shoves, taking "
            f"{m['recovery_time_s']:.2f} s on average"
            + (
                f"; {m['non_recovery_count']} never recovered"
                if m["non_recovery_count"]
                else ""
            )
        )

    lines.append(
        f"    peak motor torque reached {m['peak_torque_frac_wheel'] * 100:.0f}% of "
        f"the wheel motors' rated limit and "
        f"{m['peak_torque_frac_leg'] * 100:.0f}% of the leg motors'"
    )

    if score_result["violations"]:
        lines.append(
            "    limits exceeded: " + ", ".join(score_result["violations"])
        )
    else:
        lines.append("    no limit was exceeded")

    per_speed = ", ".join(
        f"{speed} m/s: {value:.3f}" for speed, value in score_result["per_speed"].items()
    )
    lines.append("")
    lines.append(f"Score: {score_result['score']:.3f} out of 1.000 ({per_speed})")
    return "\n".join(lines)
