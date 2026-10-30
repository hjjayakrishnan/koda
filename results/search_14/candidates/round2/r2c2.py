import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Alternative design:
    - survival bonus when upright and wheels touching ground.
    - speed reward using exponential decay of error.
    - yaw reward using exponential decay of error.
    - pitch rate penalty (penalize fast tilting).
    - joint acceleration proxy (change in velocity approximated by current velocity).
    - torque penalty.
    - action magnitude penalty to avoid aggressive commands.
    """
    # survival bonus
    pitch = abs(state.get('pitch', 0.0))
    contact = state.get('contact', 0.0)
    survival = 0.02 if (pitch < 0.3 and contact > 0.5) else 0.0

    # speed reward (exp decay)
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = abs(cmd_speed - actual_speed)
    speed_reward = 0.1 * math.exp(-speed_err / 0.2)

    # yaw reward (exp decay)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = abs(cmd_yaw - actual_yaw)
    yaw_reward = 0.05 * math.exp(-yaw_err / 0.5)

    # pitch rate penalty
    pitch_rate = state.get('pitch_rate', 0.0)
    pitch_rate_penalty = -0.03 * pitch_rate * pitch_rate

    # joint velocity penalty (proxy for acceleration)
    joint_vel_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sum += v * v
    joint_vel_penalty = -0.00012 * joint_vel_sum

    # torque penalty
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.00018 * torque_sum

    # action magnitude penalty (discourage large commands)
    ground_cmd = abs(action.get('ground_velocity', 0.0))
    yaw_cmd = abs(action.get('yaw_velocity', 0.0))
    action_penalty = -0.005 * (ground_cmd * ground_cmd + yaw_cmd * yaw_cmd)

    return {
        'survival': survival,
        'speed_reward': speed_reward,
        'yaw_reward': yaw_reward,
        'pitch_rate_penalty': pitch_rate_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'torque_penalty': torque_penalty,
        'action_penalty': action_penalty,
    }
