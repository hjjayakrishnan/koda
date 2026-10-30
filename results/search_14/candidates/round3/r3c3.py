import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - tracking_weighted: weighted sum of speed and yaw errors (quadratic).
    - stability: reward based on low pitch_rate and yaw_rate (smooth attitude).
    - torque_smooth: penalize rapid changes in torque approximated by current torque squared.
    - wheel_slip: penalize difference between wheel speed and ground velocity (slip).
    - alive_bonus: small constant for each step the robot survives.
    """
    # tracking (weighted quadratic)
    sp_err = state.get('commanded_speed', 0.0) - state.get('ground_velocity', 0.0)
    y_err = state.get('commanded_yaw_rate', 0.0) - state.get('yaw_rate', 0.0)
    tracking_weighted = -(1.2 * sp_err * sp_err + 0.4 * y_err * y_err)

    # stability (penalize high rates)
    pitch_rate = state.get('pitch_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    stability = -0.05 * (pitch_rate * pitch_rate + yaw_rate * yaw_rate)

    # torque smoothness (use current torque as proxy)
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_smooth = -0.0001 * torque_sum

    # wheel slip: difference between wheel speed and ground velocity
    left_wheel = state.get('left_wheel_velocity', 0.0)
    right_wheel = state.get('right_wheel_velocity', 0.0)
    avg_wheel = (left_wheel + right_wheel) * 0.5
    ground_vel = state.get('ground_velocity', 0.0)
    slip = (avg_wheel - ground_vel) ** 2
    wheel_slip = -0.02 * slip

    # alive bonus
    alive_bonus = 0.001

    return {
        'tracking_weighted': tracking_weighted,
        'stability': stability,
        'torque_smooth': torque_smooth,
        'wheel_slip': wheel_slip,
        'alive_bonus': alive_bonus,
    }
