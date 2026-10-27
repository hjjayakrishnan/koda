# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate fault: passes Gate 1 and then behaves badly. It pays for being in
# contact with the floor and for being level, and penalises any motion at all,
# so the best thing the robot can do is nothing. The commanded speed is never
# read. Exercises the "survived Gate 1, graded badly" path.
import math


def compute_reward(state, action):
    return {
        "on_the_floor": 2.0 * state["contact"],
        "level": math.exp(-10.0 * state["pitch"] ** 2),
        "stillness": -2.0 * abs(state["ground_velocity"]),
    }
