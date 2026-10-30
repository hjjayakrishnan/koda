import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward based on exponential shaping:
    - exp_balance: high reward when pitch is near zero (Gaussian).
    - exp_speed: reward for forward velocity aligned with command (cosine-like).
    - yaw_align: reward for yaw angle staying near zero (prevents spinning).
    - action_smooth: penalizes large changes in action (not stored, so approximate with action magnitude).
    """
    pitch = state.get('pitch', 0.0)
    # Gaussian reward, sigma = 0.2 rad (~11 deg)
    exp_balance = math.exp(-(pitch * pitch) / (2 * 0.2 * 0.2))

    ground_vel = state.get('ground_velocity', 0.0)
    cmd_speed = state.get('commanded_speed', 0.0)
    # Reward proportional to overlap; max when velocities match direction and magnitude
    # Use a simple normalized dot product (since 1D)
    if abs(cmd_speed) > 1e-3:
        exp_speed = math.exp(-((ground_vel - cmd_speed) ** 2) / (2 * 0.1 * 0.1))
    else:
        # When no forward command, reward low speed magnitude
        exp_speed = math.exp(-(ground_vel * ground_vel) / (2 * 0.05 * 0.05))

    yaw = state.get('yaw', 0.0)
    # Keep yaw near zero (straight ahead)
    yaw_align = math.exp(-(yaw * yaw) / (2 * 0.2 * 0.2))

    # Action magnitude penalty (encourage small commands)
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    action_smooth = -(ground_cmd * ground_cmd + yaw_cmd * yaw_cmd) * 0.005

    return {
        'exp_balance': exp_balance,
        'exp_speed': exp_speed,
        'yaw_align': yaw_align,
        'action_smooth': action_smooth,
    }
