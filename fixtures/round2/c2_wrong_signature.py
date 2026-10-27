# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate fault: three parameters instead of two. A believable slip once the
# prompt is long. Exercises Gate 1's signature check.
import math


def compute_reward(state, action, info):
    return {
        "upright": math.exp(-9.0 * state["pitch"] ** 2),
        "speed": -abs(state["ground_velocity"] - state["commanded_speed"]),
    }
