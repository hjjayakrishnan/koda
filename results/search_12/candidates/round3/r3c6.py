import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Minimalist yet effective:
    - upright_cosine: cosine of pitch (as in baseline) but scaled.
    - speed_error_l1: L1 penalty on speed error (less sensitive to outliers).
    - yaw_alignment_cosine: cosine of yaw (rewards facing forward).
    - joint_movement_cost: sum of absolute joint velocities (encourages calm motion).
    - torque_cost: sum of absolute torques.
    - contact_step: small reward each step with contact.
    """
    # upright cosine
    pitch = state.get('pitch', 0.0)
    upright_cosine = math.cos(pitch)  # [-1,1]

    # L1 speed error
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_error_l1 = -0.2 * abs(speed_err)

    # yaw alignment cosine (prefer yaw=0)
    yaw = state.get('yaw', 0.0)
    yaw_alignment_cosine = math.cos(yaw)  # 1 when facing forward

    # joint movement cost (L1 velocity)
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += abs(v)
    joint_movement_cost = -0.001 * joint_vel_sum

    # torque cost (L1)
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += abs(t)
    torque_cost = -0.0005 * torque_sum

    # contact step reward
    contact = state.get('contact', 0.0)
    contact_step = 0.005 * contact

    return {
        'upright_cosine': upright_cosine,
        'speed_error_l1': speed_error_l1,
        'yaw_alignment_cosine': yaw_alignment_cosine,
        'joint_movement_cost': joint_movement_cost,
        'torque_cost': torque_cost,
        'contact_step': contact_step,
    }
