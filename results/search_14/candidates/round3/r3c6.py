import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - survival: constant small reward for being alive.
    - fall_penalty: large negative reward if pitch exceeds threshold.
    - speed_ratio: reward proportional to ratio of actual to commanded speed (capped).
    - yaw_ratio: same for yaw rate.
    - joint_sync: reward for matching left/right hip and knee angles (symmetry).
    - torque_diff: penalize difference between left and right total torque (balance).
    """
    # survival
    survival = 0.002

    # fall penalty if robot leans too far
    pitch = state.get('pitch', 0.0)
    fall_penalty = 0.0
    if abs(pitch) > 0.3:  # ~17 degrees
        fall_penalty = -1.0

    # speed ratio (actual/commanded), avoid division by zero
    cmd_s = state.get('commanded_speed', 0.0)
    act_s = state.get('ground_velocity', 0.0)
    if abs(cmd_s) > 1e-3:
        speed_ratio = act_s / cmd_s
        # reward close to 1, penalize deviation
        speed_ratio_term = -0.5 * (speed_ratio - 1.0) ** 2
    else:
        # when no command, reward small forward motion
        speed_ratio_term = 0.01 * act_s

    # yaw ratio similarly
    cmd_y = state.get('commanded_yaw_rate', 0.0)
    act_y = state.get('yaw_rate', 0.0)
    if abs(cmd_y) > 1e-3:
        yaw_ratio = act_y / cmd_y
        yaw_ratio_term = -0.3 * (yaw_ratio - 1.0) ** 2
    else:
        yaw_ratio_term = 0.005 * act_y

    # joint sync: reward left/right similarity for hip and knee
    sync_pen = 0.0
    for joint in ('hip', 'knee'):
        l = state.get(f'left_{joint}_position', 0.0)
        r = state.get(f'right_{joint}_position', 0.0)
        sync_pen += (l - r) ** 2
    joint_sync = -0.02 * sync_pen

    # torque diff: penalize imbalance in total torque left vs right
    left_torque = sum(state.get(f'left_{j}_torque', '') ] ) # placeholder - fix below

Oops I made mistake while constructing last candidate due to incomplete code block. Let's rewrite final candidate correctly. We'll discard the erroneous attempt and write a clean version.

I'll produce final candidate now. Ensure proper syntax.

Proceed to final answer.
