import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward based on exponential shaping:
    - exp_speed: exponential reward for matching commanded speed.
    - exp_upright: exponential reward for being upright.
    - exp_yaw: exponential reward for low yaw rate.
    - jerk_penalty: approximates jerk via change in wheel acceleration (proxy using wheel velocity change from previous step encoded in state via wheel_position delta? Not available → use wheel_velocity squared as proxy).
    - torque_penalty: as before.
    - contact_bonus: as before.
    Note: Since we lack memory, we approximate smoothness by penalizing high wheel velocity.
    """
    # exponential speed reward
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_diff = cmd_speed - actual_speed
    exp_speed = math.exp(-0.5 * speed_diff * speed_diff)  # max 1 at zero error

    # exponential upright reward
    pitch = state.get('pitch', 0.0)
    exp_upright = math.exp(- (pitch ** 2) / (2 * 0.3 ** 2))  # sigma=0.3 rad

    # exponential yaw rate reward (prefer low yaw rate)
    yaw_rate = state.get('yaw_rate', 0.0)
    exp_yaw = math.exp(- (yaw_rate ** 2) / (2 * 0.5 ** 2))

    # smoothness proxy: penalize high wheel velocities (both wheels)
    left_wheel_v = state.get('left_wheel_velocity', 0.0)
    right_wheel_v = state.get('right_wheel_velocity', 0.0)
    wheel_vel_penalty = -0.001 * (left_wheel_v * left_wheel_v + right_wheel_v * right_wheel_v)

    # torque effort
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.0002 * torque_sum

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.01 * contact

    return {
        'exp_speed': exp_speed,
        'exp_upright': exp_upright,
        'exp_yaw': exp_yaw,
        'wheel_vel_penalty': wheel_vel_penalty,
        'torque_penalty': torque_penalty,
        'contact_bonus': contact_bonus,
    }
