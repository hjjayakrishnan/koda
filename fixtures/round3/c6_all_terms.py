# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: everything that scored well in rounds 1 and 2, in one function.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    return {
        "upright": 1.5 * math.exp(-12.0 * state["pitch"] ** 2),
        "speed_tracking": 1.2 * math.exp(-6.0 * speed_error * speed_error),
        "heading_held": 0.4 * math.exp(-4.0 * state["yaw"] ** 2),
        "pitch_damping": -0.03 * abs(state["pitch_rate"]),
        "wheel_effort": -0.04
        * (abs(state["left_wheel_torque"]) + abs(state["right_wheel_torque"])),
        "alive": 0.9 * state["contact"],
    }
