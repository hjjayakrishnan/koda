import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward emphasizing minimal control effort and upright stance."""
    r = {}

    # Upright stance: reward near-zero pitch using a narrow Gaussian
    sigma_pitch = 0.2  # radians
    r['upright_gauss'] = math.exp(-0.5 * (state['pitch'] / sigma_pitch) ** 2)

    # Forward speed reward: piecewise linear tolerance band
    tol = 0.05  # m/s tolerance around command
    speed_err = state['ground_velocity'] - state['commanded_speed']
    if abs(speed_err) <= tol:
        r['speed_bonus'] = 1.0
    else:
        r['speed_bonus'] = -0.5 * (abs(speed_err) - tol)

    # Effort cost: sum of absolute torques (L1 norm) to discourage large commands
    effort = (abs(state['left_hip_torque']) + abs(state['right_hip_torque']) +
              abs(state['left_knee_torque']) + abs(state['right_knee_torque']) +
              abs(state['left_wheel_torque']) + abs(state['right_wheel_torque']))
    r['effort_l1'] = -0.002 * effort

    # Contact encouragement (small positive when touching)
    r['contact'] = 0.3 if state['contact'] > 0.5 else 0.0

    return r
