# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 2: adds a heading term, because round 1 reported heading drift.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    yaw_error = abs(state["yaw"])
    return {
        "upright": math.exp(-9.0 * state["pitch"] ** 2),
        "speed_tracking": math.exp(-5.0 * speed_error * speed_error),
        "heading_held": math.exp(-3.0 * yaw_error * yaw_error),
        "alive": 0.4 * state["contact"],
    }
