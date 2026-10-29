def compute_reward(state: dict, action: dict) -> dict:
    """Linear‑quadratic hybrid: reward forward progress, penalise pitch/yaw
    deviation, joint torque, and encourage persistent foot contact."""
    r = {}

    # Forward progress reward: proportional to ground velocity when moving
    # in the desired direction (sign matches commanded_speed)
    prog = state['ground_velocity'] * state['commanded_speed']
    r['progress'] = 0.5 * prog if prog > 0.0 else 0.0   # only reward forward motion

    # Pitch penalty: quadratic within ±0.3 rad, linear beyond
    p = abs(state['pitch'])
    if p <= 0.3:
        r['pitch'] = -1.0 * (p*p)          # small quadratic cost
    else:
        r['pitch'] = -0.3*0.3 - 2.0*(p-0.3) # continue linearly steeper

    # Yaw penalty: keep robot pointing forward (yaw≈0)
    y = abs(state['yaw'])
    r['yaw'] = -0.5 * (y*y)                # gentle quadratic

    # Contact reward: shaped by how long contact persists (using contact flag)
    r['contact'] = 0.8 if state['contact'] > 0.5 else -0.5

    # Torque cost: weighted sum (legs cheaper than wheels)
    torque_cost = (
        0.01*(state['left_hip_torque']**2 + state['right_hip_torque']**2 +
              state['left_knee_torque']**2 + state['right_knee_torque']**2) +
        0.001*(state['left_wheel_torque']**2 + state['right_wheel_torque']**2)
    )
    r['torque'] = -torque_cost

    return r
