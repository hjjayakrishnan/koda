# Hand-written fixture. Stands in for one candidate of model output so the loop can
# be built and debugged with no API call. NOT written by Nemotron.
# Deliberate fault: 1/error is infinite when the error is zero, which happens on
# the very first step when the robot is still and the command is zero.
# Exercises Gate 1's dead_nan path.
def compute_reward(state, action):
    speed_error = abs(state["ground_velocity"] - state["commanded_speed"])
    tracking = 1.0 / speed_error if speed_error > 0.0 else float("inf")
    return {
        "tracking": tracking,
        "upright": -abs(state["pitch"]),
    }
