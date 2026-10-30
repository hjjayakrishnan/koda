import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sparse reward with shaping hints:
    - big_reward: +2 if the robot is both well balanced (|pitch|<0.04) and
                  tracking speed within 0.05 m/s.
    - small_penalty: -0.01 per step to encourage shorter episodes unless big reward achieved.
    - joint_limit_soft: gentle penalty when joints approach reasonable limits (±2.5 rad for hips/knees,
                        ±10 rad for wheels) to avoid mechanical strain.
    """
    pitch = state.get('pitch', 0.0)
    ground_vel = state.get('ground_velocity', 0.0)
    cmd_speed = state.get('commanded_speed', 0.0)

    balance_ok = abs(pitch) < 0.04
    speed_ok = abs(ground_vel - cmd_speed) < 0.05
    big_reward = 2.0 if (balance_ok and speed_ok) else 0.0

    small_penalty = -0.01  # step cost

    # Joint limit soft penalty
    limit_penalty = 0.0
    # hips/knees limit ~2.5 rad
    for side in ('left', 'right'):
        for joint in ('hip', 'knee'):
            pos = state.get(f'{side}_{joint}_position', 0.0)
            if abs(pos) > 2.5:
                excess = abs(pos) - 2.5
                limit_penalty -= 0.05 * excess * excess
    # wheels limit ~10 rad (just to avoid huge windup)
    for side in ('left', 'right'):
        pos = state.get(f'{side}_wheel_position', 0.0)
        if abs(pos) > 10.0:
            excess = abs(pos) - 10.0
            limit_penalty -= 0.02 * excess * excess

    return {
        'big_reward': big_reward,
        'step_cost': small_penalty,
        'joint_limit_soft': limit_penalty,
    }
