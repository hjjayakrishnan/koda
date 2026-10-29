import math

def compute_reward(state: dict, action: dict) -> dict:
    """Hybrid reward: exponential decay for large errors, linear bonus for small errors."""
    r = {}
    # Pitch: linear bonus inside ±0.1 rad, exponential penalty outside
    p_err = abs(state['pitch'])
    if p_err <= 0.1:
        r['pitch'] = 0.2 * (0.1 - p_err)  # max 0.02 when perfectly upright
    else:
        r['pitch'] = -0.5 * (math.exp(p_err - 0.1) - 1.0)

    # Speed: similar treatment with ±0.1 m/s deadzone
    s_err = abs(state['ground_velocity'] - state['commanded_speed'])
    if s_err <= 0.1:
        r['speed'] = 0.15 * (0.1 - s_err)
    else:
        r['speed'] = -0.6 * (math.exp(s_err - 0.1) - 1.0)

    # Yaw alignment: reward pointing roughly forward (|yaw| < 0.2 rad)
    yaw_abs = abs(state['yaw'])
    if yaw_abs <= 0.2:
        r['yaw_align'] = 0.1 * (0.2 - yaw_abs)
    else:
        r['yaw_align'] = -0.2 * (yaw_abs - 0.2)

    # Effort: quadratic torque penalty
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['effort'] = -0.0003 * torque_sq

    # Contact: strong incentive to keep wheels on ground
    r['contact'] = 1.0 if state['contact'] > 0.5 else -1.0

    return r
