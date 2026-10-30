import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - speed_exp: exponential decay reward for speed tracking.
    - yaw_exp: exponential decay reward for yaw tracking.
    - pitch_penalty: quadratic penalty for large pitch (falling).
    - joint_energy: penalize sum of squared joint velocities (effort).
    - contact_keep: small reward for maintaining contact.
    - progress: reward forward movement regardless of command.
    """
    # speed tracking using exp(-error^2)
    cmd_s = state.get('commanded_speed', 0.0)
    act_s = state.get('ground_velocity', 0.0)
    sp_err = cmd_s - act_s
    speed_exp = math.exp(- (sp_err * sp_err) / 0.5) - 1.0  # zero at perfect match, negative otherwise

    # yaw tracking similarly
    cmd_y = state.get('commanded_yaw_rate', 0.0)
    act_y = state.get('yaw_rate', 0.0)
    y_err = cmd_y - act_y
    yaw_exp = math.exp(- (y_err * y_err) / 0.5) - 1.0

    # pitch penalty (quadratic)
    pitch = state.get('pitch', 0.0)
    pitch_penalty = -0.8 * pitch * pitch

    # joint velocity energy
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_energy = -0.00005 * joint_vel_sum

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_keep = 0.02 * contact

    # progress reward (forward movement)
    progress = state.get('ground_velocity', 0.0) * 0.01

    return {
        'speed_exp': speed_exp,
        'yaw_exp': yaw_exp,
        'pitch_penalty': pitch_penalty,
        'joint_energy': joint_energy,
        'contact_keep': contact_keep,
        'progress': progress,
    }
