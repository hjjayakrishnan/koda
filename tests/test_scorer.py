# SPDX-License-Identifier: Apache-2.0
"""R0 -- red-team the grader before anything is scored with it.

    python3 tests/test_scorer.py

Two kinds of check:

  1. The grader must not be able to see reward values or term names. Checked
     against the source text, not by trusting discipline.
  2. Three deliberate cheats must be caught. Each one is expressed as the
     *physical rollout* that cheat produces, because what the grader sees is a
     rollout -- it never sees the reward that caused it. The reward files
     themselves are in `cheaters/`.

Training the three cheats for real is the full test and takes about an hour of
CPU. This file tests the grader's logic, which is the part that has to be right
before any scored run.
"""

import inspect
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koda import eval_set, scorer  # noqa: E402
from koda import state as state_mod  # noqa: E402

PASSED = 0


def check(label, condition, detail=""):
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        sys.exit(1)


def episode(**over):
    """A healthy 20-second episode. Override what a given cheat changes."""
    base = {
        "kind": "task",
        "seed": 11,
        "commanded_speed": 0.6,
        "commanded_yaw_rate": 0.0,
        "episode_seconds": 20.0,
        "steps_survived": 4000,
        "step_limit": 4000,
        "mean_speed_error": 0.03,
        "net_displacement_m": 11.4,
        "heading_error_deg": 2.0,
        "max_pitch_excursion_deg": 3.0,
        "closest_joint_margin_deg": 60.0,
        "peak_torque_nm": {"leg": 4.0, "wheel": 1.2},
        "rms_torque_nm": {"leg": 1.5, "wheel": 0.5},
        "fell": False,
    }
    base.update(over)
    return base


def rollout(episodes):
    return {"schema_version": "2.0", "episodes": episodes}


def code_identifiers(module):
    """Every name the module's CODE uses -- docstrings and comments excluded.

    Checking the raw source text is useless here: this grader's own docstring
    explains at length what it must never see, so the word "reward" appears in
    it on purpose. What matters is whether the executable code ever touches a
    reward concept.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    # Drop every docstring before looking at anything.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def boundary():
    print("The grader cannot see reward values or term names")
    parameters = list(inspect.signature(scorer.score).parameters)
    check("score() takes rollout and an id, nothing else",
          parameters == ["rollout", "candidate_id"], str(parameters))

    names = code_identifiers(scorer)
    for word in ("reward", "term", "fitness", "compute_reward"):
        hits = sorted(n for n in names if word in n.lower())
        # `active_measurements` and `measurements` are the grader's own output
        # names and have nothing to do with reward terms.
        hits = [h for h in hits if "measurement" not in h.lower()]
        check(f"no identifier or literal in the CODE contains '{word}'",
              not hits, str(hits))
    for term in ("upright", "speed_tracking", "alive", "tracking", "posture",
                 "stay_upright", "energy"):
        check(f"no hard-coded reward term name '{term}'",
              not any(term in n.lower() for n in names))


def healthy_scores_well():
    print("A healthy rollout scores well and trips nothing")
    result = scorer.score(rollout([episode()]), "healthy")
    check(f"score {result['score']:.3f} is above 0.7", result["score"] > 0.7)
    check("no violations", result["violations"] == [], str(result["violations"]))
    check("scorer_sha recorded", len(result["scorer_sha"]) == 7)
    check("active_measurements recorded",
          result["active_measurements"] == list(scorer.ACTIVE_MEASUREMENTS))


def cheat_collapse():
    print("Cheat 1 -- collapse and farm the contact bonus")
    # Lying on the floor: the episode ends early and it goes nowhere.
    caught = scorer.score(
        rollout([episode(steps_survived=180, episode_seconds=0.9,
                         net_displacement_m=0.05, mean_speed_error=0.58,
                         max_pitch_excursion_deg=58.0, fell=True)]),
        "collapse",
    )
    check(f"scores low ({caught['score']:.3f} < 0.25)", caught["score"] < 0.25)
    check("flagged fell_over", "fell_over" in caught["violations"],
          str(caught["violations"]))
    check("survival_frac is small",
          caught["measurements"]["survival_frac"] < 0.1)


def cheat_spin():
    print("Cheat 2 -- spin the wheels, go nowhere")
    # Upright the whole time, wheels turning, body stationary.
    caught = scorer.score(
        rollout([episode(net_displacement_m=0.12, mean_speed_error=0.59)]),
        "spin",
    )
    check(f"scores low ({caught['score']:.3f} < 0.3)", caught["score"] < 0.3)
    check("displacement_frac is near zero",
          abs(caught["measurements"]["displacement_frac"]) < 0.05,
          str(caught["measurements"]["displacement_frac"]))
    # The trap this closes: translating in a circle at exactly the commanded
    # speed. Speed error is perfect and it still goes nowhere.
    circling = scorer.score(
        rollout([episode(mean_speed_error=0.001, net_displacement_m=0.3,
                         heading_error_deg=38.0)]),
        "circling",
    )
    check(f"circling at the right speed still scores low "
          f"({circling['score']:.3f} < 0.35)", circling["score"] < 0.35)


def cheat_torque():
    print("Cheat 3 -- slam the motors to farm an effort bonus")
    caught = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 4.0, "wheel": 2.4})]),
        "torque_slam",
    )
    check("flagged wheel_torque_exceeded",
          "wheel_torque_exceeded" in caught["violations"],
          str(caught["violations"]))
    check(f"peak_torque_frac_wheel reported as "
          f"{caught['measurements']['peak_torque_frac_wheel']}",
          caught["measurements"]["peak_torque_frac_wheel"] > 1.4)

    legs = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 17.5, "wheel": 1.2})]),
        "leg_slam",
    )
    check("flagged leg_torque_exceeded",
          "leg_torque_exceeded" in legs["violations"], str(legs["violations"]))


def saturation_is_a_violation():
    print("Reaching 100% of rated torque is a violation (user rule, 2026-10-27)")
    # Upkie's own shipped controller measured exactly 1.7000 Nm on 2026-10-26.
    baseline = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 1.45, "wheel": 1.7})]),
        "upkie_shipped_controller",
    )
    check("exactly at the rating is a violation",
          "wheel_torque_exceeded" in baseline["violations"],
          str(baseline["violations"]))
    check("peak_torque_frac_wheel reads exactly 1.0",
          baseline["measurements"]["peak_torque_frac_wheel"] == 1.0)
    check("the hand-engineered baseline is NOT exempted",
          baseline["violations"] != [])

    # The floating-point question: clipped-to-exactly-1.7 and a rounding error
    # below it must be treated the same.
    rounded = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 1.45, "wheel": 1.69999})]),
        "rounded",
    )
    check("a rounding error below the limit counts the same as touching it",
          "wheel_torque_exceeded" in rounded["violations"],
          str(rounded["violations"]))
    check("the tolerance is written down, not implicit",
          scorer.TORQUE_TOLERANCE == 0.001)

    under = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 1.45, "wheel": 1.64})]),
        "under",
    )
    check("comfortably under the rating trips nothing",
          under["violations"] == [], str(under["violations"]))


def saturated_step_frac():
    print("How LONG a motor sat at the limit is recorded, not just whether")
    brief = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 1.45, "wheel": 1.7},
                         saturated_steps={"leg": 0, "wheel": 12})]),
        "brief_touch",
    )
    pinned = scorer.score(
        rollout([episode(peak_torque_nm={"leg": 1.45, "wheel": 1.7},
                         saturated_steps={"leg": 0, "wheel": 4000})]),
        "pinned",
    )
    check("touching the ceiling briefly reads near zero",
          brief["measurements"]["saturated_step_frac_wheel"] == 0.003,
          str(brief["measurements"]["saturated_step_frac_wheel"]))
    check("sitting at the limit the whole episode reads 1.0",
          pinned["measurements"]["saturated_step_frac_wheel"] == 1.0,
          str(pinned["measurements"]["saturated_step_frac_wheel"]))
    check("both still violate -- the flag alone cannot tell them apart",
          "wheel_torque_exceeded" in brief["violations"]
          and "wheel_torque_exceeded" in pinned["violations"])
    check("so the fraction is what carries the meaning",
          pinned["measurements"]["saturated_step_frac_wheel"]
          > brief["measurements"]["saturated_step_frac_wheel"] * 100)

    # A rollout from before this field existed must not get an invented number.
    old = scorer.score(rollout([episode()]), "old_rollout")
    check("a rollout without the field reports None, not a guess",
          old["measurements"]["saturated_step_frac_wheel"] is None)


def stationary_episode():
    print("A stationary episode does not produce a nonsense number")
    # This is the PD baseline's case. The old grader returned -93300.0 here.
    result = scorer.score(
        rollout([episode(commanded_speed=0.0, net_displacement_m=-0.093,
                         mean_speed_error=0.02)]),
        "stationary",
    )
    check("displacement_frac is None, not a huge number",
          result["measurements"]["displacement_frac"] is None,
          str(result["measurements"]["displacement_frac"]))
    check(f"it still gets a sensible score ({result['score']:.3f})",
          0.0 <= result["score"] <= 1.0)
    check("no violations from standing still",
          result["violations"] == [], str(result["violations"]))


def blind_spots_are_preserved():
    print("The deliberate blind spots are still blind")
    names = code_identifiers(scorer)
    for forbidden in ("action_rate", "jerk", "vertical_velocity", "lin_vel_z",
                      "dof_acc", "energy", "smoothness", "vibration"):
        hits = sorted(n for n in names if forbidden in n.lower())
        check(f"the code measures nothing called '{forbidden}'",
              not hits, str(hits))
    # Two rollouts identical except one shakes itself apart. Same score: that
    # is the blind spot, and it is deliberate.
    smooth = scorer.score(rollout([episode()]), "smooth")
    chattering = scorer.score(
        rollout([episode(rms_torque_nm={"leg": 9.0, "wheel": 1.5})]),
        "chattering",
    )
    check("a chattering policy scores the same as a smooth one",
          smooth["score"] == chattering["score"],
          f"{smooth['score']} vs {chattering['score']}")


def push_handling():
    print("Push episodes: non-recoveries are counted, never averaged in")
    # `disturbed=True` on every push episode here: the trainer always writes
    # this field, and a robot cannot have recovered without first having been
    # knocked off upright. These fixtures predated the field. Since defect 9
    # the scorer gates recovery on it, so leaving it out described a rollout
    # that cannot physically occur.
    episodes = [episode()]
    episodes.append(episode(kind="push", disturbed=True, recovered=True,
                            recovery_time_s=1.2))
    episodes.append(episode(kind="push", disturbed=True, recovered=True,
                            recovery_time_s=1.8))
    episodes.append(episode(kind="push", disturbed=True, recovered=False,
                            recovery_time_s=20.0))
    result = scorer.score(rollout(episodes), "pushed")
    check("recovery_time_s averages only the recoveries (1.5)",
          result["measurements"]["recovery_time_s"] == 1.5,
          str(result["measurements"]["recovery_time_s"]))
    check("the non-recovery is counted separately",
          result["measurements"]["non_recovery_count"] == 1)

    none_recovered = [episode()] + [
        episode(kind="push", disturbed=True, recovered=False,
                recovery_time_s=20.0)
    ]
    result = scorer.score(rollout(none_recovered), "never")
    check("never recovering is flagged",
          "never_recovered_from_push" in result["violations"])
    check("recovery_time_s is None when nothing recovered",
          result["measurements"]["recovery_time_s"] is None)


def falling_early_does_not_hide_bad_driving():
    """The flaw this caught: metrics averaged over surviving steps only.

    `mean_speed_error` and `displacement_frac` were both computed over the part
    of the episode the robot stayed upright, so falling early deleted the
    evidence of bad driving. Measured on real data at a 0.9 m/s command, r1c3
    scored a speed error of 0.212 over 92 control steps before falling, against
    the hand-engineered MPC controller's 0.338 over 4,000 -- the robot that
    crashed looked like the better driver.

    Every other check in this file passed with that in place.
    """
    limit = eval_set.step_limit(200.0) if hasattr(eval_set, "step_limit") else 4000

    # Same driving quality, one survives, one lunges and falls at 10%.
    survivor = episode(commanded_speed=0.9, steps_survived=limit, step_limit=limit,
                       mean_speed_error=0.30, net_displacement_m=0.9 * (limit / 200.0) * 0.6,
                       episode_seconds=limit / 200.0, fell=False)
    luncher = episode(commanded_speed=0.9, steps_survived=limit // 10, step_limit=limit,
                      mean_speed_error=0.21, net_displacement_m=0.37,
                      episode_seconds=(limit // 10) / 200.0, fell=True)

    s_sur = scorer.score(rollout([survivor]), "survivor")
    s_lun = scorer.score(rollout([luncher]), "luncher")

    check(
        "a policy that falls at 10% does not beat one that survives on speed error",
        s_lun["measurements"]["mean_speed_error"]
        > s_sur["measurements"]["mean_speed_error"],
        f"luncher {s_lun['measurements']['mean_speed_error']} "
        f"vs survivor {s_sur['measurements']['mean_speed_error']}",
    )
    check(
        "the raw while-upright figure is still reported, so the two are auditable",
        s_lun["measurements"]["mean_speed_error_while_up"] < 0.25,
        str(s_lun["measurements"].get("mean_speed_error_while_up")),
    )
    check(
        "and it scores lower overall",
        s_lun["score"] < s_sur["score"],
        f"{s_lun['score']} vs {s_sur['score']}",
    )


def tracking_has_no_dead_zone():
    """The bug the old grader had, and nothing here caught it.

    A linear `max(0, 1 - error/0.3)` scored every mean error at or above
    0.3 m/s as exactly zero, so 0.32 m/s and 2.0 m/s were identical across 55%
    of the score. Every other test in this file passed with that in place.
    """
    import math

    errors = [0.0, 0.1, 0.2, 0.3, 0.321, 0.4, 0.5, 0.75, 1.0]
    parts = [math.exp(-(e * e) / scorer.SPEED_SIGMA) for e in errors]

    # strictly decreasing -- a worse tracker always scores strictly lower
    for a_, b_ in zip(parts, parts[1:]):
        check(
            f"tracking strictly decreasing past {a_:.3f}",
            b_ < a_,
            f"{b_} is not < {a_}",
        )

    # and never collapses to a flat zero anywhere a candidate might land
    check(
        "no dead zone at the baseline's own error (0.321 m/s)",
        math.exp(-(0.321**2) / scorer.SPEED_SIGMA) > 0.5,
        "the baseline's tracking term is near zero again",
    )
    check(
        "still discriminating at 1.0 m/s error",
        math.exp(-(1.0**2) / scorer.SPEED_SIGMA) > 0.0,
        "flat at 1.0 m/s",
    )


def tracking_matches_the_published_form():
    """It must be legged_gym's metric, not one of ours.

    `exp(-error**2 / tracking_sigma)` with sigma = 0.25 is the standard in
    legged locomotion RL (Isaac Lab's `track_lin_vel_xy_exp`). If someone
    retunes sigma or changes the shape, this fails and they have to say so.
    """
    import math

    check("tracking_sigma is the standard 0.25", scorer.SPEED_SIGMA == 0.25)
    for error, expected in ((0.0, 1.0), (0.5, math.exp(-1.0)), (1.0, math.exp(-4.0))):
        got = math.exp(-(error * error) / scorer.SPEED_SIGMA)
        check(
            f"exp form exact at error {error}",
            abs(got - expected) < 1e-12,
            f"{got} != {expected}",
        )


def standing_still_cannot_be_bought_back():
    """A robot that goes nowhere scores zero, however smooth and upright."""
    print("Going nowhere cannot be paid for with anything else")

    # Perfectly upright, perfectly smooth, never leaves the spot.
    still = rollout([
        episode(seed=s, commanded_speed=sp, net_displacement_m=0.0,
                mean_speed_error=sp, saturated_steps={"leg": 0, "wheel": 0})
        for s in (11, 23, 37) for sp in (0.3, 0.6, 0.9)
    ])
    verdict = scorer.score(still, "stationary")
    check("a robot that never moves scores exactly 0",
          verdict["score"] == 0.0, f"got {verdict['score']}")

    # Drifting BACKWARD is not better than standing still.
    backward = rollout([
        episode(seed=s, commanded_speed=sp, net_displacement_m=-0.5,
                mean_speed_error=sp, saturated_steps={"leg": 0, "wheel": 0})
        for s in (11, 23, 37) for sp in (0.3, 0.6, 0.9)
    ])
    check("a robot that drifts backward scores exactly 0",
          scorer.score(backward, "backward")["score"] == 0.0)


def thrashing_is_punished():
    """Same job done, but one slams the wheels at their limit the whole time."""
    print("Sitting on the torque limit costs real score")

    def at_limit(fraction):
        return rollout([
            episode(seed=s, commanded_speed=sp,
                    net_displacement_m=sp * 20.0 * 0.5,
                    mean_speed_error=0.3,
                    saturated_steps={"leg": 0,
                                     "wheel": int(4000 * fraction)})
            for s in (11, 23, 37) for sp in (0.3, 0.6, 0.9)
        ])

    smooth = scorer.score(at_limit(0.05), "smooth")["score"]
    rough = scorer.score(at_limit(0.95), "rough")["score"]
    check("identical driving, 95% saturated scores below 5% saturated",
          rough < smooth, f"{rough} vs {smooth}")
    check("the gap is large, not cosmetic", rough < smooth * 0.2,
          f"{rough} vs {smooth}")
    check("saturation is one of the scored measurements",
          "saturated_step_frac_wheel" in scorer.ACTIVE_MEASUREMENTS)


def geometric_mean_never_inflates():
    """AM-GM: the geometric mean cannot exceed the average it replaced."""
    print("The geometric mean never scores higher than the old average")
    worst = 0.0
    for speed_error in (0.05, 0.3, 0.6, 1.0):
        for fraction in (0.1, 0.3, 0.6, 1.0):
            roll = rollout([
                episode(seed=s, commanded_speed=sp,
                        net_displacement_m=sp * 20.0 * fraction,
                        mean_speed_error=speed_error,
                        heading_error_deg=0.0,
                        saturated_steps={"leg": 0, "wheel": 0})
                for s in (11, 23, 37) for sp in (0.3, 0.6, 0.9)
            ])
            got = scorer.score(roll, "x")["score"]
            speed_part = math.exp(-(speed_error ** 2) / scorer.SPEED_SIGMA)
            average = (scorer.WEIGHTS["speed"] * speed_part
                       + scorer.WEIGHTS["displacement"] * min(1.0, fraction))
            worst = max(worst, got - average)
            check(f"error {speed_error}, distance {fraction}: "
                  f"{got:.4f} <= {average:.4f}", got <= average + 1e-9)
    check("no case inflated the score", worst <= 1e-9, f"worst excess {worst}")


def recovery_needs_an_actual_disturbance():
    """Defect 9: the shove flag must not fire when nothing shoved the robot."""
    print("never_recovered_from_push needs something to have recovered FROM")

    def with_push(**over):
        base = dict(kind="push", commanded_speed=0.6,
                    saturated_steps={"leg": 0, "wheel": 0})
        base.update(over)
        return [episode(seed=s, net_displacement_m=6.0, **base)
                for s in (11, 23, 37, 53, 71)]

    task = [episode(seed=s, commanded_speed=sp, net_displacement_m=sp * 20.0 * 0.5,
                    saturated_steps={"leg": 0, "wheel": 0})
            for s in (11, 23, 37) for sp in (0.3, 0.6, 0.9)]

    # Nothing ever crossed the pitch threshold -- there was no shove at all.
    calm = scorer.score(rollout(task + with_push(disturbed=False,
                                                 recovered=False)), "calm")
    check("no disturbance means no never_recovered_from_push violation",
          "never_recovered_from_push" not in calm["violations"],
          str(calm["violations"]))
    check("non_recovery_count is 0, not the number of push episodes",
          calm["measurements"]["non_recovery_count"] == 0,
          str(calm["measurements"]["non_recovery_count"]))
    check("disturbed_count reports 0",
          calm["measurements"]["disturbed_count"] == 0)

    # Genuinely knocked over and never came back: the flag MUST still fire.
    knocked = scorer.score(rollout(task + with_push(disturbed=True,
                                                    recovered=False)), "knocked")
    check("a real disturbance with no recovery still violates",
          "never_recovered_from_push" in knocked["violations"],
          str(knocked["violations"]))
    check("non_recovery_count counts all 5", 
          knocked["measurements"]["non_recovery_count"] == 5)

    # Knocked over and recovered: no violation.
    ok = scorer.score(rollout(task + with_push(disturbed=True, recovered=True,
                                               recovery_time_s=1.4)), "ok")
    check("disturbed then recovered does not violate",
          "never_recovered_from_push" not in ok["violations"],
          str(ok["violations"]))

    check("the three cases do not change the score",
          calm["score"] == knocked["score"] == ok["score"],
          f"{calm['score']} {knocked['score']} {ok['score']}")


def real_cheaters_parse():
    print("The three cheat files exist and are valid reward functions")
    from koda import gate1

    files = sorted(Path("cheaters").glob("*.py"))
    check("three cheat files", len(files) == 3, str(files))
    for path in files:
        verdict = gate1.check(path.read_text(), path.stem)
        check(f"{path.name} passes Gate 1 (a cheat must be valid code)",
              verdict["ok"], str(verdict["error"]))


if __name__ == "__main__":
    boundary()
    healthy_scores_well()
    cheat_collapse()
    cheat_spin()
    cheat_torque()
    saturation_is_a_violation()
    saturated_step_frac()
    stationary_episode()
    blind_spots_are_preserved()
    push_handling()
    falling_early_does_not_hide_bad_driving()
    tracking_has_no_dead_zone()
    tracking_matches_the_published_form()
    standing_still_cannot_be_bought_back()
    thrashing_is_punished()
    geometric_mean_never_inflates()
    recovery_needs_an_actual_disturbance()
    real_cheaters_parse()
    print(f"\n{PASSED} checks passed")
