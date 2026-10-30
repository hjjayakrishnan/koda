import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - forward_progress: displacement this step (ground_velocity * dt).
    - upright_cos2: Squared cosine of pitch (strong penalty for tilt).
    - yaw_rate_penalty: Penalize non-zero yaw rate unless commanded.
    - action_magnitude: Penalty on the magnitude of the action vector.
    - joint_energy: Penalty on squared joint velocities (effort).
    - contact_bonus: Bonus for wheel‑ground contact.
    """
    # Forward progress (metres per step)
    dt = state.get('dt', 0.01)
    ground_vel = state.get('ground_velocity', 0.0)
    forward_progress = ground_vel * dt  # reward for moving forward

    # Uprightness: use cos^2 to heavily penalize tilt
    pitch = state.get('pitch', 0.0)
    upright_cos2 = math.cos(pitch) * math.cos(pitch)  # [0,1]

    # Yaw rate: only reward if matches commanded yaw rate, otherwise penalize
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - yaw_rate
    yaw_rate_penalty = -0.4 * yaw_err * yaw_err  # quadratic penalty

    # Action magnitude penalty (encourages mild commands)
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    action_mag = ground_cmd * ground_cmd + yaw_cmd * yaw_cmd
    action_penalty = -0.0008 * action_mag

    # Joint energy (velocity squared)
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_energy = -0.00012 * joint_vel_sum

    # Contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.01 * contact

    return {
        'forward_progress': forward_progress,
        'upright_cos2': upright_cos2,
        'yaw_rate_penalty': yaw_rate_penalty,
        'action_penalty': action_penalty,
        'joint_energy': joint_energy,
        'contact_bonus': contact_bonus,
    }
