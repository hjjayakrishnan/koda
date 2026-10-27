# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: a hard pitch band, flat reward inside it, steep cost outside.
import math


def compute_reward(state, action):
    pitch = abs(state["pitch"])
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    inside_band = 1.0 if pitch < 0.08 else 0.0
    return {
        "in_band": 1.6 * inside_band,
        "out_of_band": -3.0 * max(0.0, pitch - 0.08),
        "speed_tracking": 1.1 * math.exp(-6.0 * speed_error * speed_error),
        "alive": 0.7 * state["contact"],
    }
