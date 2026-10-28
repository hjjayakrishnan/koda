import math

def compute_reward(state: dict, action: dict) -> dict:
    # Balance reward: penalize pitch away from upright
    pitch_err = state['pitch']
    balance = -8.0 * pitch_err * pitch_err

    # Speed tracking reward: compare actual ground velocity to commanded
    vel_err = state['ground_velocity'] - action['ground_velocity']
    speed = -4.0 * vel_err * vel_err

    # Small survival bonus to encourage staying alive
    survive = 0.5

    return {
        'balance': balance,
        'speed': speed,
        'survive': survive
    }
