def compute_reward(state: dict, action: dict) -> dict:
    """Sparse‑dense hybrid: give a large positive reward when the robot
    is upright, moving forward, and in contact; otherwise give small
    shaping gradients to guide learning."""
    r = {}

    # Dense shaping: negative quadratic costs for pitch and speed error
    p = state['pitch']
    r['pitch_shape'] = -0.8 * (p * p)

    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_shape'] = -0.5 * (speed_err * speed_err)

    # Sparse bonus: +1.0 only when all three conditions are met
    upright = abs(state['pitch']) < 0.2          # ~11°
    moving_forward = state['ground_velocity'] * state['commanded_speed'] > 0.0
    contact_ok = state['contact'] > 0.5
    if upright and moving_forward and contact_ok:
        r['sparse_bonus'] = 1.0
    else:
        r['sparse_bonus'] = 0.0

    # Contact maintenance: small reward for staying in touch
    r['contact_keep'] = 0.2 if state['contact'] > 0.5 else -0.1

    # Effort penalty: linear in absolute joint torques (legs) + wheel torques
    leg_torque = (
        abs(state['left_hip_torque']) + abs(state['right_hip_torque']) +
        abs(state['left_knee_torque']) + abs(state['right_knee_torque'])
    )
    wheel_torque = (
        abs(state['left_wheel_torque']) + abs(state['right_wheel_torque'])
    )
    r['effort'] = -0.01 * leg_torque - 0.002 * wheel_torque

    return r
</response>
