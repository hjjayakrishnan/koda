# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate state: half right. It pays for ground covered and nothing else, so
# balance is not in the objective at all.
def compute_reward(state, action):
    return {
        "distance": 0.4 * state["ground_position"],
        "pitch_rate_penalty": -0.05 * abs(state["pitch_rate"]),
    }
