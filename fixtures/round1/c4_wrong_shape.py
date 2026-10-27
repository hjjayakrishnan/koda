# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate fault: returns a list of term values instead of a dict of named
# terms. Exercises Gate 1's wrong-return-type path.
import math


def compute_reward(state, action):
    return [
        math.exp(-6.0 * state["pitch"] ** 2),
        -abs(state["ground_velocity"] - state["commanded_speed"]),
        state["contact"],
    ]
