# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate fault: the file does not parse. Missing colon on the def line.
# Exercises Gate 1's dead_compile path.
import math


def compute_reward(state, action)
    return {
        "upright": 1.0 - abs(state["pitch"]),
        "speed": -abs(state["ground_velocity"] - state["commanded_speed"]),
    }
