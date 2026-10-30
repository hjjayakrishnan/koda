import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Potential‑based shaping:
    Define a desired feature vector (pitch, pitch_rate, ground_velocity error,
    yaw_rate error). Reward is negative squared Mahalanobis distance with hand‑picked
    weights, plus a survival bonus.
    """
    # Errors
    pitch_err = state.get('pitch', 0.0)
    pitch_rate_err = state.get('pitch_rate', 0.0)
    speed_err = state.get('ground_velocity', 0.0) - state.get('commanded_speed', 0.0)
    yaw_rate_err = state.get('yaw_rate', 0.0) - state.get('commanded_yaw_rate', 0.0)

    # Weights (tuned intuitively)
    w_pitch = 10.0
    w_pitch_rate = 2.0
    w_speed = 5.0
    w_yaw_rate = 2.0

    # Quadratic cost
    cost = (w_pitch * pitch_err * pitch_err +
            w_pitch_rate * pitch_rate_err * pitch_rate_err +
            w_speed * speed_err * speed_err +
            w_yaw_rate * yaw_rate_err * yaw_rate_err)

    # Survival bonus (encourage longer episodes)
    survive = 0.1

    return {
        'potential': -cost,
        'survive': survive,
    }
