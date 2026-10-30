import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - speed_track: negative quadratic error between commanded and actual forward speed.
    - yaw_track: negative quadratic error between commanded and actual yaw rate.
    - upright_gauss: Gaussian reward for small pitch (max 1 at upright).
    - joint_vel_penalty: penalizes high joint angular velocities (smooth motion).
    - torque_penalty: penalizes squared joint torques (energy use).
    - contact_bonus: small reward for maintaining wheel-ground contact.
    """
    # speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_track = -0.5 * speed_err * speed_err

    # yaw tracking
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - actual_yaw
    yaw_track = -0.3 * yaw_err * yaw_err   # weight tuned lower than speed

    # uprightness via Gaussian
    pitch = state.get('pitch', 0.0)
    upright_gauss = math.exp(- (pitch ** 2) / (2 * 0.2 ** 2))  # sigma=0.2 rad (~11°)

    # joint velocity smoothness penalty
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_vel_penalty = -0.0001 * joint_vel_sum

    # torque effort penalty
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.0002 * torque_sum

    # contact encouragement
    contact = state.get('contact', 0.0)
    contact_bonus = 0.01 * contact   # 0.01 when both wheels touch, else 0

    return {
        'speed_track': speed_track,
        'yaw_track': yaw_track,
        'upright_gauss': upright_gauss,
        'joint_vel_penalty': joint_vel_penalty,
        'torque_penalty': torque_penalty,
        'contact_bonus': contact_bonus,
    }
