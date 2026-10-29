import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward based on rates (stability) and forward progress, with torque regularization."""
    r = {}
    # Stability: low pitch_rate and yaw_rate are good
    r['pitch_rate'] = -0.5 * math.tanh(abs(state['pitch_rate']))
    r['yaw_rate']   = -0.3 * math.tanh(abs(state['yaw_rate']))

    # Forward progress: reward moving in direction of commanded speed
    # Use dot product of actual ground velocity vector (approx forward) and commanded speed.
    # Assume heading aligned with forward when yaw≈0; penalize yaw misalignment.
    forward_alignment = math.cos(state['yaw'])  # 1 when facing forward, -1 when opposite
    progress = state['ground_velocity'] * forward_alignment
    r['progress'] = 0.4 * progress  # positive when moving forward correctly

    # Speed tracking: small L2 penalty to keep speed close to command
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_track'] = -0.2 * (speed_err * speed_err)

    # Effort: torque cost
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['effort'] = -0.0002 * torque_sq

    # Contact bonus
    r['contact'] = 0.5 if state['contact'] > 0.5 else -0.5

    return r
