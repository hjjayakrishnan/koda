# SPDX-License-Identifier: Apache-2.0
"""The reward-function input contract.

A reward function is handed two flat dictionaries of plain floats: `state` and
`action`. This module is the single place that says what keys those dictionaries
have, what the units are, and what the sign conventions are.

Everything that produces a state dict uses this module:

  * `koda.trainer_stub`     -- Stage A, made-up numbers, no simulator
  * `koda.trainer_pybullet` -- Stage B, real PyBullet rollouts

so the two cannot drift apart.
"""

import math

# --------------------------------------------------------------------------
# The state dictionary
# --------------------------------------------------------------------------
# (key, unit, meaning). This list is what `koda.prompt` puts in front of the
# model. Nothing else about the state dictionary is given to the model.

JOINTS = (
    "left_hip",
    "left_knee",
    "left_wheel",
    "right_hip",
    "right_knee",
    "right_wheel",
)

BASE_FIELDS = (
    ("dt", "s", "length of one control step"),
    ("pitch", "rad", "base pitch from vertical, positive leaning forward"),
    ("pitch_rate", "rad/s", "base pitch angular velocity"),
    ("ground_position", "m", "position of the average wheel contact point"),
    ("ground_velocity", "m/s", "velocity of the average wheel contact point"),
    ("yaw", "rad", "base yaw angle, zero at the start of the episode"),
    ("yaw_rate", "rad/s", "base yaw angular velocity"),
    ("commanded_speed", "m/s", "the forward speed the robot was asked for"),
    ("commanded_yaw_rate", "rad/s", "the yaw rate the robot was asked for"),
    ("contact", "0 or 1", "1.0 when at least one wheel touches the floor"),
)

JOINT_FIELDS = (
    ("{joint}_position", "rad", "joint angle"),
    ("{joint}_velocity", "rad/s", "joint angular velocity"),
    ("{joint}_torque", "Nm", "torque applied by the joint's motor"),
)

ACTION_FIELDS = (
    ("ground_velocity", "m/s", "commanded forward velocity of the wheel contact point"),
    ("yaw_velocity", "rad/s", "commanded yaw velocity"),
)


def state_keys() -> tuple:
    """Every key a state dictionary contains, in a fixed order."""
    keys = [name for name, _unit, _meaning in BASE_FIELDS]
    for joint in JOINTS:
        for pattern, _unit, _meaning in JOINT_FIELDS:
            keys.append(pattern.format(joint=joint))
    return tuple(keys)


def action_keys() -> tuple:
    """Every key an action dictionary contains, in a fixed order."""
    return tuple(name for name, _unit, _meaning in ACTION_FIELDS)


def field_table() -> tuple:
    """The (key, unit, meaning) rows for the state dictionary, expanded."""
    rows = list(BASE_FIELDS)
    for joint in JOINTS:
        for pattern, unit, meaning in JOINT_FIELDS:
            rows.append((pattern.format(joint=joint), unit, f"{joint}: {meaning}"))
    return tuple(rows)


# --------------------------------------------------------------------------
# Physical limits, read from upkie_description/urdf/upkie.urdf
# --------------------------------------------------------------------------
# These are NOT given to the model. The grader thresholds on them, so putting
# them in the prompt would hand the model a grader threshold.

LEG_JOINTS = ("left_hip", "left_knee", "right_hip", "right_knee")
WHEEL_JOINTS = ("left_wheel", "right_wheel")

LEG_TORQUE_LIMIT_NM = 16.0  # mjbots qdd100, URDF effort="16.0"
WHEEL_TORQUE_LIMIT_NM = 1.7  # mjbots mj5208, URDF effort="1.7"

FALL_PITCH_RAD = 1.0  # upkie's own fall-detection threshold


# --------------------------------------------------------------------------
# Building a state dictionary from upkie's environment (Stage B)
# --------------------------------------------------------------------------


def state_from_upkie(
    observation,
    spine_observation: dict,
    commanded_speed: float,
    commanded_yaw_rate: float,
    dt: float,
) -> dict:
    """Flatten upkie's two objects into one state dictionary.

    `observation` is UpkieGyropod's 6-number vector:
        [ground_position, pitch, yaw, ground_velocity, pitch_rate, yaw_rate]
    `spine_observation` is the nested dictionary found at info["spine_observation"].
    """
    servo = spine_observation.get("servo", {})
    floor_contact = spine_observation.get("floor_contact", {})

    state = {
        "dt": float(dt),
        "pitch": float(observation[1]),
        "pitch_rate": float(observation[4]),
        "ground_position": float(observation[0]),
        "ground_velocity": float(observation[3]),
        "yaw": float(observation[2]),
        "yaw_rate": float(observation[5]),
        "commanded_speed": float(commanded_speed),
        "commanded_yaw_rate": float(commanded_yaw_rate),
        "contact": 1.0 if floor_contact.get("contact", False) else 0.0,
    }
    for joint in JOINTS:
        joint_obs = servo.get(joint, {})
        state[f"{joint}_position"] = float(joint_obs.get("position", 0.0))
        state[f"{joint}_velocity"] = float(joint_obs.get("velocity", 0.0))
        state[f"{joint}_torque"] = float(joint_obs.get("torque", 0.0))
    return state


# --------------------------------------------------------------------------
# A made-up trajectory (Stage A and Gate 1)
# --------------------------------------------------------------------------

DT = 0.01  # 100 Hz control loop


def zero_state(commanded_speed: float = 0.0) -> dict:
    """A state dictionary with every number at zero, standing on the floor."""
    state = {key: 0.0 for key in state_keys()}
    state["dt"] = DT
    state["contact"] = 1.0
    state["commanded_speed"] = commanded_speed
    return state


def fake_trajectory(
    steps: int,
    commanded_speed: float,
    wobble: float = 0.08,
    falls_at: int | None = None,
    moves: bool = True,
) -> list:
    """A plausible but entirely made-up sequence of state dictionaries.

    Used by Gate 1 to call a candidate reward function, and by the stub trainer
    to produce per-term curves. No simulator is involved: the pitch is a decaying
    sine wave, the speed ramps toward the command, and the torques follow the
    pitch. `falls_at` tips the robot over from that step onward.
    """
    trajectory = []
    for step in range(steps):
        t = step * DT
        decay = math.exp(-t / 2.0)
        fallen = falls_at is not None and step >= falls_at

        if fallen:
            pitch = min(FALL_PITCH_RAD + 0.2, 0.3 + 0.9 * (step - falls_at) * DT)
            pitch_rate = 1.4
            ground_velocity = 0.0
        else:
            pitch = wobble * decay * math.sin(6.0 * t)
            pitch_rate = wobble * decay * 6.0 * math.cos(6.0 * t)
            ground_velocity = (
                commanded_speed * (1.0 - math.exp(-t / 0.8)) if moves else 0.0
            )

        state = zero_state(commanded_speed)
        state["pitch"] = pitch
        state["pitch_rate"] = pitch_rate
        state["ground_position"] = (
            commanded_speed * max(0.0, t - 0.8) if moves else 0.0
        )
        state["ground_velocity"] = ground_velocity
        state["yaw"] = 0.02 * math.sin(0.7 * t)
        state["yaw_rate"] = 0.014 * math.cos(0.7 * t)
        state["contact"] = 0.0 if fallen else 1.0

        wheel_torque = 0.9 * pitch + 0.25 * ground_velocity
        for joint in WHEEL_JOINTS:
            state[f"{joint}_torque"] = wheel_torque
            state[f"{joint}_velocity"] = ground_velocity / 0.06  # 6 cm wheel radius
        for joint in LEG_JOINTS:
            state[f"{joint}_torque"] = 2.5 + 3.0 * abs(pitch)
            state[f"{joint}_position"] = 0.3 if "knee" in joint else -0.15

        trajectory.append(state)
    return trajectory


def fake_action(state: dict) -> dict:
    """The action that goes with a made-up state."""
    return {
        "ground_velocity": state["commanded_speed"],
        "yaw_velocity": state["commanded_yaw_rate"],
    }
