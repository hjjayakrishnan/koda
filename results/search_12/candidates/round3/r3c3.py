import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sparse‑style reward:
    - stability_bonus: large reward when pitch within ±0.15 rad AND contact=1.
    - speed_match: tapered reward for speed error using a soft‑clipped quadratic.
    - yaw_align: reward for having yaw close to zero (facing forward).
    - joint_limit_penalty: grows as joint angles approach ±1.5 rad (≈86°).
    - control_effort: penalty for magnitude of the action commands.
    """
    # stability region
    pitch = state.get('pitch', 0.0)
    contact = state.get('contact', 0.0)
    if abs(pitch) < 0.15 and contact > 0.5:
        stability_bonus = 1.0
    else:
        stability_bonus = 0.0

    # speed match with soft cap (less punishing for large errors)
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_match = -0.3 * math.tanh(speed_err * speed_err)  # bounded [-0.3,0]

    # yaw alignment (prefer zero yaw)
    yaw = state.get('yaw', 0.0)
    yaw_align = math.exp(- (yaw ** 2) / (2 * 0.5 ** 2))  # sigma=0.5 rad

    # joint limit penalty (avoid extremes)
    limit = 1.5  # rad
    joint_pen = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            ang = state.get(f'{side}_{joint}_position', 0.0)
            # distance beyond soft limit
            excess = max(0.0, abs(ang) - 0.8 * limit)  # start penalizing after 80%
            joint_pen += excess * excess
    joint_limit_penalty = -0.02 * joint_pen

    # control effort (magnitude of commanded velocities)
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    control_effort = -0.05 * (abs(ground_cmd) + abs(yaw_cmd))

    return {
        'stability_bonus': stability_bonus,
        'speed_match': speed_match,
        'yaw_align': yaw_align,
        'joint_limit_penalty': joint_limit_penalty,
        'control_effort': control_effort,
    }
