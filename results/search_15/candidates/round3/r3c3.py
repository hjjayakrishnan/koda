import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - speed_exp: Exponential decay reward for speed error (exp(-|error|/scale)).
    - yaw_exp: Same for yaw rate.
    - pitch_pen: Quadratic penalty on pitch (keeps robot near upright).
    - jerk_proxy: Penalty on pitch_rate and yaw_rate (discourages rapid shaking).
    - motor_use: Penalty on absolute joint torques (L1 norm) to save energy.
    - contact_bonus: Small reward for contact.
    """
    # Speed tracking with exponential shaping
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = abs(cmd_speed - actual_speed)
    speed_exp = math.exp(-speed_err / 0.3)  # max 1 when error=0
    speed_reward = speed_exp - 1.0  # shift so zero error gives 0, negative otherwise

    # Yaw tracking similarly
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = abs(cmd_yaw - actual_yaw)
    yaw_exp = math.exp(-yaw_err / 0.5)
    yaw_reward = yaw_exp - 1.0

    # Pitch penalty (quadratic)
    pitch = state.get('pitch', 0.0)
    pitch_pen = -0.8 * pitch * pitch  # stronger penalty for tilt

    # Jerk proxy: penalize angular rates of base
    pitch_rate = state.get('pitch_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    jerk_proxy = -0.02 * (pitch_rate * pitch_rate + yaw_rate * yaw_rate)

    # Motor use: L1 norm of torques (encourages smaller torques)
    torque_l1 = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_l1 += abs(t)
    motor_use = -0.001 * torque_l1

    # Contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.008 * contact

    return {
        'speed_exp': speed_reward,
        'yaw_exp': yaw_reward,
        'pitch_pen': pitch_pen,
        'jerk_proxy': jerk_proxy,
        'motor_use': motor_use,
        'contact_bonus': contact_bonus,
    }
