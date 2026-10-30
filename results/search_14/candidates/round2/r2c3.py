import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Energy‑aware reward:
    - power penalty: torque * angular velocity summed over joints.
    - speed tracking: negative quadratic error.
    - yaw tracking: negative quadratic error.
    - upright Gaussian.
    - joint velocity penalty.
    - contact bonus.
    """
    # power consumption approximation
    power = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            power += abs(t * v)
    power_penalty = -0.0005 * power

    # speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    sp_err = cmd_speed - act_speed
    speed_track = -0.5 * sp_err * sp_err

    # yaw tracking
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    y_err = cmd_yaw - act_yaw
    yaw_track = -0.3 * y_err * y_err

    # upright Gaussian (sigma = 0.25 rad)
    pitch = state.get('pitch', 0.0)
    upright_gauss = math.exp(- (pitch * pitch) / (2 * 0.25 * 0.25))

    # joint velocity penalty
    jv2 = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            jv2 += v * v
    joint_vel_penalty = -0.00008 * jv2

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.01 * contact

    return {
        'power_penalty': power_penalty,
        'speed_track': speed_track,
        'yaw_track': yaw_track,
        'upright_gauss': upright_gauss,
        'joint_vel_penalty': joint_vel_penalty,
        'contact_bonus': contact_bonus,
    }
