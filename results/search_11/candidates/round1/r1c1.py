import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward focused on upright posture and speed tracking."""
    r = {}

    # Penalize leaning away from upright (absolute pitch)
    pitch_err = abs(state['pitch'])
    r['pitch_penalty'] = -4.0 * pitch_err ** 2

    # Reward maintaining contact with the ground (helps stability)
    r['contact_bonus'] = 1.0 if state['contact'] > 0.5 else -1.0

    # Track commanded forward speed
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_tracking'] = -3.0 * speed_err ** 2

    # Small penalty for rapid pitching motions (damping)
    r['pitch_rate_penalty'] = -0.5 * state['pitch_rate'] ** 2

    return r
