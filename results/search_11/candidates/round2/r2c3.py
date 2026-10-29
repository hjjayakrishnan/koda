import math

def compute_reward(state: dict, action: dict) -> dict:
    """Piecewise‑linear penalties with dead zones; encourages staying within safe bounds."""
    r = {}
    # Pitch deadzone ±0.05 rad (~3°); outside, linear penalty
    p = abs(state['pitch'])
    if p <= 0.05:
        r['pitch'] = 0.0
    else:
        r['pitch'] = -1.0 * (p - 0.05)

    # Speed deadzone ±0.05 m/s
    s_err = state['ground_velocity'] - state['commanded_speed']
    s = abs(s_err)
    if s <= 0.05:
        r['speed'] = 0.0
    else:
        r['speed'] = -0.8 * (s - 0.05)

    # Joint limit safety: penalize approaching ±1.0 rad (~57°) for hip/knee
    joint_names = ['left_hip_position', 'right_hip_position',
                   'left_knee_position', 'right_knee_position']
    joint_pen = 0.0
    for jn in joint_names:
        ang = abs(state.get(jn, 0.0))
        if ang > 0.8:  # start penalizing after 0.8 rad
            joint_pen += -(ang - 0.8) * 2.0
    r['joint_safety'] = joint_pen

    # Effort: small quadratic cost
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['effort'] = -0.0003 * torque_sq

    # Contact encouragement
    r['contact'] = 0.6 if state['contact'] > 0.5 else -0.4

    return r
