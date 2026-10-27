# SPDX-License-Identifier: Apache-2.0
"""Strip the robot's identity out of the prompt text before it is sent.

Upkie's reward functions are public on GitHub and Codeberg, so the model may
have memorised one. Telling a model "do not copy the published reward" cannot be
verified and raises the salience of the thing being avoided, so Koda removes the
identity from the input instead.

What this hides: the robot's name, its author's name, and its motor part numbers.
What it cannot hide: the morphology. Two legs, two wheels, six actuated joints, a
balancing task. A capable model will work out "wheeled balancing biped" -- it just
will not know which one, and copying the published reward requires knowing that.

The simulator, the URDF and the physics are untouched. This changes prompt text
and nothing else.
"""

import re

# Longest first: "UpkieGyropod" must be replaced before "Upkie".
SUBSTITUTIONS = (
    ("UpkieBaseVelocity", "RobotBaseVelocity"),
    ("UpkieGyropod", "RobotEnv"),
    ("UpkieServos", "RobotServos"),
    ("UpkiePendulum", "RobotPendulum"),
    ("UpkieEnv", "RobotEnv"),
    ("upkie_description", "robot_description"),
    ("upkie_base_velocity", "robot_base_velocity"),
    ("upkie_gyropod", "robot_env"),
    ("upkie_env", "robot_env"),
    ("gyropod", "wheeled_pendulum"),
    ("Gyropod", "WheeledPendulum"),
    ("qdd100", "actuator_a"),
    ("mj5208", "actuator_b"),
    ("mjbots", "the motor vendor"),
    ("scaron.info", "example.invalid"),
    ("scaron", "the author"),
    ("Upkie", "the robot"),
    ("upkie", "robot"),
)

# Anything from this list surviving in the prompt is a leak. Checked, not assumed.
BANNED = ("upkie", "qdd100", "mj5208", "mjbots", "gyropod", "scaron", "wheel_tire")


# A last pass, case-insensitive, for spellings the ordered list above misses --
# "UPKIE" in a heading, say. Without it an odd capitalisation reaches
# `sanitize_or_raise`, which refuses to send the prompt at all.
LAST_PASS = (
    ("upkie", "robot"),
    ("qdd100", "actuator_a"),
    ("mj5208", "actuator_b"),
    ("mjbots", "the motor vendor"),
    ("gyropod", "wheeled_pendulum"),
    ("scaron", "the author"),
    ("wheel_tire", "wheel_rubber"),
)


def sanitize(text: str) -> str:
    """Replace every identifying string. Returns the prompt as it will be sent."""
    for before, after in SUBSTITUTIONS:
        text = text.replace(before, after)
    for before, after in LAST_PASS:
        text = re.sub(before, after, text, flags=re.IGNORECASE)
    return text


def leaks(text: str) -> list:
    """Identifying strings still present. Empty list means the prompt is clean."""
    lowered = text.lower()
    return [word for word in BANNED if word in lowered]


def sanitize_or_raise(text: str) -> str:
    """Sanitize, then refuse to return a prompt that still names the robot."""
    clean = sanitize(text)
    remaining = leaks(clean)
    if remaining:
        raise ValueError(
            "prompt still identifies the robot after sanitizing: "
            + ", ".join(remaining)
        )
    return clean
