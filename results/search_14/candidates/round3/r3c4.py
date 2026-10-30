import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - forward_gain: reward proportional to forward velocity when moving in desired direction.
    - yaw_align: reward for yaw alignment using dot product of forward vectors.
    - pitch_penalty_huber: Huber loss on pitch to reduce sensitivity to large angles.
    - joint_limit_avoid: soft penalty when joint angles approach ±0.5 rad.
    - motor_usage: penalize absolute torque (linear) to encourage low effort.
    - contact_stick: reward for maintaining contact (binary).
    """
    # forward gain: encourage moving forward if command positive, else backward
    cmd_speed = state.get('commanded_speed', 0.0)
    ground_vel = state.get('ground_velocity', 0.0)
    forward_gain = 0.03 * cmd_speed * ground_vel  # positive when signs match

    # yaw align: reward when yaw rate matches command direction
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_align = 0.02 * cmd_yaw * yaw_rate

    # pitch Huber loss (delta=0.2 rad)
    pitch = state.get('pitch', 0.0)
    abs_pitch = abs(pitch)
    delta = 0.2
    if abs_pitch <= delta:
        pitch_penalty_huber = -0.5 * pitch * pitch
    else:
        pitch_penalty_huber = -delta * (abs_pitch - 0.5 * delta)
    # note: already negative; we return as is

    # joint limit avoidance (soft barrier around +/-0.5 rad)
    limit_pen = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            ang = state.get(f'{side}_{joint}_position', 0.0)
            # penalty grows as angle approaches limit
            over = abs(abs(ang) - 0.5)
            if over < 0:
                limit_pen += (-over) * (-over)  # square of violation
    joint_limit_avoid = -0.01 * limit_pen

    # motor usage (linear torque)
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += abs(t)
    motor_usage = -0.005 * torque_sum

    # contact stick
    contact = state.get('contact', 0.0)
    contact_stick = 0.015 * contact

    return {
        'forward_gain': forward_gain,
        'yaw_align': yaw_align,
        'pitch_penalty_huber': pitch_penalty_huber,
        'joint_limit_avoid': joint_limit_avoid,
        'motor_usage': motor_usage,
        'contact_stick': contact_stick,
    }
