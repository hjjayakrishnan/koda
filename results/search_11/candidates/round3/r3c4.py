def compute_reward(state: dict, action: dict) -> dict:
    """Energy‑based reward: reward kinetic energy of forward motion,
    penalise potential energy from pitching, and minimise control effort."""
    r = {}

    # Forward kinetic energy proxy: 0.5 * mass * v^2 (mass omitted → scaling)
    v = state['ground_velocity']
    r['forward_ke'] = 0.1 * (v * v) if v > 0.0 else 0.0   # only reward forward motion

    # Potential energy from pitch: higher when leaned forward/backward
    p = state['pitch']
    r['potential_pitch'] = -0.05 * (p * p)   # penalises deviation from upright

    # Yaw kinetic energy: reward small yaw rate (stability)
    y_rate = state['yaw_rate']
    r['yaw_stability'] = -0.02 * (y_rate * y_rate)

    # Contact reward: encourage keeping at least one wheel on ground
    r['contact'] = 0.3 if state['contact'] > 0.5 else -0.2

    # Control effort: squared norm of action (what we command)
    act_norm = (action.get('ground_velocity', 0.0)**2 +
                action.get('yaw_velocity', 0.0)**2)
    r['action_effort'] = -0.001 * act_norm

    return r
