import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - speed_huber: Huber loss for speed tracking (less sensitive to outliers).
    - yaw_huber: Huber loss for yaw rate tracking.
    - upright_cos: Cosine-based upright reward (max 1 at pitch=0).
    - joint_vel_penalty: Penalty on squared joint velocities.
    - torque_penalty: Penalty on squared joint torques.
    - contact_bonus: Small bonus for maintaining ground contact.
    """
    # Helper for Huber loss
    def huber(x, delta=1.0):
        abs_x = abs(x)
        if abs_x <= delta:
            return 0.5 * x * x
        else:
            return delta * (abs_x - 0.5 * delta)

    # Speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_huber = -huber(speed_err, delta=0.5)  # negative because we want to minimise error

    # Yaw tracking
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - actual_yaw
    yaw_huber = -huber(yaw_err, delta=0.2)

    # Uprightness via cosine (cos(0)=1, cos(±π/2)=0)
    pitch = state.get('pitch', 0.0)
    upright_cos = math.cos(pitch)  # ranges [-1,1]; we shift to [0,1] later
    upright_cos = max(0.0, upright_cos)  # only reward when not overly tilted

    # Joint velocity penalty
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_vel_penalty = -0.00015 * joint_vel_sum

    # Torque penalty
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.00025 * torque_sum

    # Contact encouragement
    contact = state.get('contact', 0.0)
    contact_bonus = 0.012 * contact

    return {
        'speed_huber': speed_huber,
        'yaw_huber': yaw_huber,
        'upright_cos': upright_cos,
        'joint_vel_penalty': joint_vel_penalty,
        'torque_penalty': torque_penalty,
        'contact_bonus': contact_bonus,
    }
