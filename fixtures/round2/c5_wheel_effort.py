# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 2: a different idea. Pays for tracking and charges for wheel effort.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    wheel_effort = abs(state["left_wheel_torque"]) + abs(state["right_wheel_torque"])
    return {
        "upright": math.exp(-8.0 * state["pitch"] ** 2),
        "speed_tracking": math.exp(-6.0 * speed_error * speed_error),
        "wheel_effort": -0.08 * wheel_effort,
        "alive": 0.5 * state["contact"],
    }
