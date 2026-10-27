# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 3: pays for ground covered per step at the commanded rate, instead of
# for matching the instantaneous speed.
import math


def compute_reward(state, action):
    progress = state["ground_velocity"] * state["dt"]
    wanted = state["commanded_speed"] * state["dt"]
    shortfall = abs(progress - wanted) / max(1e-6, state["dt"])
    return {
        "upright": 1.4 * math.exp(-11.0 * state["pitch"] ** 2),
        "progress": 1.0 * math.exp(-5.0 * shortfall * shortfall),
        "alive": 0.8 * state["contact"],
        "pitch_damping": -0.025 * abs(state["pitch_rate"]),
    }
