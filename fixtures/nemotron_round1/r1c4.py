import math

def compute_reward(state: dict, action: dict) -> dict:
    # Yaw tracking: reward matching commanded yaw rate
    yaw_err = state['yaw_rate'] - action['yaw_velocity']
    yaw_reward = -2.0 * yaw_err * yaw_err

    # Pitch penalty (keep upright)
    pitch_pen = -5.0 * state['pitch'] * state['pitch']

    # Forward progress bonus: reward positive ground velocity
    forward = state['ground_velocity'] if state['ground_velocity'] > 0.0 else 0.0
    forward = 0.3 * forward

    return {
        'yaw': yaw_reward,
        'pitch': pitch_pen,
        'forward': forward
    }
