# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 2: a different idea. Holds a crouch, since the legs are not otherwise
# in the objective.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    crouch_error = abs(state["left_knee_position"] - 0.3) + abs(
        state["right_knee_position"] - 0.3
    )
    return {
        "upright": math.exp(-9.0 * state["pitch"] ** 2),
        "speed_tracking": math.exp(-5.0 * speed_error * speed_error),
        "posture": -0.3 * crouch_error,
        "alive": 0.5 * state["contact"],
    }
