import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Survival‑based reward with explicit fall detection.
    - alive_bonus: constant for staying upright (contact > 0).
    - fall_penalty: large negative when contact lost.
    - speed_match: reward for achieving commanded speed within a tolerance.
    - joint_limit_penalty: grows as any joint approaches ±π/2 rad (~90°).
    """
    # contact determines if robot is still standing
    contact = state.get('contact', 0.0)
    alive_bonus = 0.02 * contact          # 0.02 when touching ground, else 0
    fall_penalty = -1.0 * (1.0 - contact) # -1 if no contact

    # speed matching: reward inside a 0.05 m/s band, linearly decreasing outside
    cmd = state.get('commanded_speed', 0.0)
    vel = state.get('ground_velocity', 0.0)
    err = abs(cmd - vel)
    if err <= 0.05:
        speed_match = 0.03                # full bonus
    else:
        speed_match = max(0.0, 0.03 - 0.1 * (err - 0.05))

    # joint limit avoidance (soft barrier)
    limit_pen = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            ang = abs(state.get(f'{side}_{joint}_position', 0.0))
            # treat π/2 ≈ 1.57 as hard limit; use quadratic penalty beyond 1.2 rad
            if ang > 1.2:
                excess = ang - 1.2
                limit_pen += excess * excess
    joint_limit_penalty = -0.02 * limit_pen

    return {
        'alive_bonus': alive_bonus,
        'fall_penalty': fall_penalty,
        'speed_match': speed_match,
        'joint_limit_penalty': joint_limit_penalty,
    }
