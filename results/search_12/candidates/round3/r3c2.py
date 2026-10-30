import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Alternative design:
    - forward_aligned: reward for moving forward in the direction the robot is facing.
    - pitch_penalty: quadratic penalty for pitch deviation.
    - yaw_penalty: quadratic penalty for yaw rate deviation from zero (encourages straight).
    - wheel_slip_penalty: penalty for mismatch between commanded wheel speed and actual wheel speed.
    - motor_effort: penalty for absolute torque*velocity (instantaneous power).
    - alive_bonus: small constant reward for each step the robot is alive.
    """
    # forward aligned velocity (projected onto body forward axis)
    yaw = state.get('yaw', 0.0)
    ground_vel = state.get('ground_velocity', 0.0)
    forward_aligned = ground_vel * math.cos(yaw)   # positive when moving forward

    # pitch penalty (quadratic)
    pitch = state.get('pitch', 0.0)
    pitch_penalty = -0.5 * pitch * pitch

    # yaw rate penalty (keep yaw rate low unless commanded)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_penalty = -0.2 * yaw_rate * yaw_rate

    # wheel slip: compare commanded ground velocity to average wheel velocity
    left_wheel_v = state.get('left_wheel_velocity', 0.0)
    right_wheel_v = state.get('right_wheel_velocity', 0.0)
    avg_wheel_v = (left_wheel_v + right_wheel_v) / 2.0
    slip_err = state.get('ground_velocity', 0.0) - avg_wheel_v
    wheel_slip_penalty = -0.4 * slip_err * slip_err

    # motor effort (instantaneous power approximation)
    power_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            tau = state.get(f'{side}_{joint}_torque', 0.0)
            vel = state.get(f'{side}_{joint}_velocity', 0.0)
            power_sum += abs(tau * vel)
    motor_effort = -0.001 * power_sum

    # alive bonus
    alive_bonus = 0.02

    return {
        'forward_aligned': forward_aligned,
        'pitch_penalty': pitch_penalty,
        'yaw_penalty': yaw_penalty,
        'wheel_slip_penalty': wheel_slip_penalty,
        'motor_effort': motor_effort,
        'alive_bonus': alive_bonus,
    }
