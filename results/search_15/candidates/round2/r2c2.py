import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Components:
    - speed_huber: Huber loss for speed tracking (less sensitive to outliers).
    - yaw_huber: Huber loss for yaw rate tracking.
    - upright_tanh: tanh-shaped reward for small pitch.
    - joint_limit: soft penalty when joints approach limits (±2.5 rad).
    - power: instantaneous power consumption estimate.
    """
    def huber(x, delta=0.5):
        abs_x = abs(x)
        if abs_x <= delta:
            return 0.5 * x * x
        else:
            return delta * (abs_x - 0.5 * delta)

    # speed tracking
    cmd_sp = state.get('commanded_speed', 0.0)
    act_sp = state.get('ground_velocity', 0.0)
    speed_err = cmd_sp - act_sp
    speed_huber = -huber(speed_err, delta=0.2)

    # yaw tracking
    cmd_yawr = state.get('commanded_yaw_rate', 0.0)
    act_yawr = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yawr - act_yawr
    yaw_huber = -huber(yaw_err, delta=0.1)

    # uprightness via tanh (bounded, smooth)
    pitch = state.get('pitch', 0.0)
    upright_tanh = math.tanh(1.0 - abs(pitch) / 0.3)  # max ~1 at zero pitch, goes to 0 at ±0.3 rad

    # joint limit avoidance (soft barrier)
    joint_limit_pen = 0.0
    limit = 2.5  # rad
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            ang = state.get(f'{side}_{joint}_position', 0.0)
            # penalty grows as angle approaches limit
            over = (abs(ang) - limit) if abs(ang) > limit else 0.0
            joint_limit_pen += over * over
    joint_limit = -0.001 * joint_limit_pen

    # power estimate: torque * angular velocity summed over actuators
    power = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            power += abs(t * v)
    power_pen = -0.0005 * power

    return {
        'speed_huber': speed_huber,
        'yaw_huber': yaw_huber,
        'upright_tanh': upright_tanh,
        'joint_limit': joint_limit,
        'power_penalty': power_pen,
    }
