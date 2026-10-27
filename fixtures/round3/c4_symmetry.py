# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: adds a left/right symmetry term on the wheels.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    wheel_mismatch = abs(state["left_wheel_velocity"] - state["right_wheel_velocity"])
    return {
        "upright": 1.4 * math.exp(-11.0 * state["pitch"] ** 2),
        "speed_tracking": 1.2 * math.exp(-6.0 * speed_error * speed_error),
        "symmetry": -0.02 * wheel_mismatch,
        "alive": 0.8 * state["contact"],
    }
