import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Minimalist reward:
    - constant step bonus for staying alive (contact & small pitch).
    - speed tracking using Huber loss (quadratic small error, linear large).
    - yaw tracking using Huber loss.
    - joint torque L1 penalty.
    - joint velocity L2 penalty.
    """
    # alive bonus
    pitch = abs(state.get('pitch', 0.0))
    contact = state.get('contact', 0.0)
    alive_bonus = 0.01 if (contact > 0.5 and pitch < 0.4) else 0.0

    # Huber loss for speed (delta = 0.2)
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - act_speed
    if abs(speed_err) <= 0.2:
        speed_loss = 0.5 * speed_err * speed_err
    else:
        speed_loss = 0.2 * (abs(speed_err) - 0.5 * 0.2)
    speed_track = -speed_loss

    # Huber loss for yaw (delta = 0.3)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - act_yaw
    if abs(yaw_err) <= 0.3:
        yaw_loss = 0.5 * yaw_err * yaw_err
    else:
        yaw_loss = 0.3 * (abs(yaw_err) - 0.5 * 0.3)
    yaw_track = -0.5 * yaw_loss  # weight lower

    # joint torque L1 penalty
    torque_l1 = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_l1 += abs(t)
    torque_penalty = -0.0004 * torque_l1

    # joint velocity L2 penalty
    jv2 = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            jv2 += v * v
    joint_vel_penalty = -0.00007 * jv2

    return {
        'alive_bonus': alive_bonus,
        'speed_track': speed_track,
        'yaw_track': yaw_track,
        'torque_penalty': torque_penalty,
        'joint_vel_penalty': joint_vel_penalty,
    }
