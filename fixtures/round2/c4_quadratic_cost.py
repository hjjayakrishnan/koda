# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Round 2: a different idea. One quadratic cost over everything, no bonuses.
def compute_reward(state, action):
    speed_error = state["ground_velocity"] - state["commanded_speed"]
    return {
        "pitch_cost": -4.0 * state["pitch"] ** 2,
        "speed_cost": -2.0 * speed_error ** 2,
        "yaw_cost": -0.5 * state["yaw"] ** 2,
        "upright_bonus": 1.0 * state["contact"],
    }
