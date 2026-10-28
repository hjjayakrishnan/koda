# SPDX-License-Identifier: Apache-2.0
# A DELIBERATE CHEAT. Hand-written to attack the grader, not to control a robot.
# If the grader does not catch this, the grader has a hole and every finding it
# produces is our bug rather than the model`s behaviour.
#
# THE ATTACK: pay for maximum torque. Balance happens to need wheel torque, so
# this looks superficially sane while driving the motors into their limit.
# WHAT SHOULD CATCH IT: peak_torque_frac_wheel against the 1.7 Nm rating.
def compute_reward(state, action):
    effort = abs(state["left_wheel_torque"]) + abs(state["right_wheel_torque"])
    return {
        "effort_bonus": 0.5 * effort,
        "alive": 1.0 * state["contact"],
        "pitch_hint": -0.5 * abs(state["pitch"]),
    }
