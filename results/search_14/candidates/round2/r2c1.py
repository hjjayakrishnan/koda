import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward terms:
    - speed_track: negative quadratic error for forward speed.
    - yaw_track: negative quadratic error for yaw rate.
    - upright: inverse quadratic reward for staying upright (max 1).
    - joint_vel_penalty: penalizes high joint angular velocities.
    - torque_penalty: penalizes squared joint torques.
    - contact_bonus: small reward for maintaining ground contact.
    """
    # speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_track = -0.4 * speed_err * speed_err

    # yaw tracking
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - actual_yaw
    yaw_track = -0.2 * yaw_err * yaw_err

    # uprightness (inverse quadratic, max=1 at pitch=0)
    pitch = state.get('pitch', 0.0)
    upright = 1.0 / (1.0 + pitch * pitch)

    # joint velocity smoothness penalty
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_vel_penalty = -0.00015 * joint_vel_sum

    # torque effort penalty
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.00025 * torque_sum

    # contact encouragement
    contact = state.get('contact', 0.0)
    contact_bonus = 0.012 * contact

    return {
        'speed_track': speed_track,
        'yaw_track': yaw_track,
        'upright': upright,
        'joint_vel_penalty': joint_vel_penalty,
        'torque_penalty': torque_penalty,
        'contact_bonus': contact_bonus,
    }
