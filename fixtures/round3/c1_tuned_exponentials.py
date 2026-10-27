# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: round 2's winner with the weights pushed harder.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    return {
        "upright": 1.5 * math.exp(-12.0 * state["pitch"] ** 2),
        "speed_tracking": 1.2 * math.exp(-6.0 * speed_error * speed_error),
        "heading_held": 0.4 * math.exp(-4.0 * state["yaw"] ** 2),
        "pitch_damping": -0.03 * abs(state["pitch_rate"]),
        "alive": 0.8 * state["contact"],
    }
