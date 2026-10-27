# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: charges for the gap between what was commanded and what the wheels
# are doing, which is the only thing in the state that looks like control effort.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    command_gap = abs(action["ground_velocity"] - state["ground_velocity"])
    return {
        "upright": 1.5 * math.exp(-12.0 * state["pitch"] ** 2),
        "speed_tracking": 1.1 * math.exp(-6.0 * speed_error * speed_error),
        "command_gap": -0.05 * command_gap,
        "alive": 0.8 * state["contact"],
    }
