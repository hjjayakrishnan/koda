# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 2: keeps round 1's winner and adds damping on the pitch rate.
import math


def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    return {
        "upright": math.exp(-10.0 * state["pitch"] ** 2),
        "speed_tracking": math.exp(-5.0 * speed_error * speed_error),
        "pitch_damping": -0.02 * abs(state["pitch_rate"]),
        "alive": 0.5 * state["contact"],
    }
