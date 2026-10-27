# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate state: healthy. Upright term, speed tracking, alive bonus.
import math


def compute_reward(state, action):
    pitch_error = abs(state["pitch"])
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    return {
        "upright": math.exp(-8.0 * pitch_error * pitch_error),
        "speed_tracking": math.exp(-4.0 * speed_error * speed_error),
        "alive": 0.5 * state["contact"],
    }
