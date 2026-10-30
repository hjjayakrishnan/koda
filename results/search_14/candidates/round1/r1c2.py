import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward focusing on forward progress and stability:
    - progress: reward proportional to forward velocity when commanded speed > 0.
    - stability_pitch: heavy penalty if pitch exceeds a safety threshold.
    - yaw_align: reward for matching commanded yaw rate with a cosine similarity.
    - jerk_proxy: penalizes large changes in wheel acceleration approximated by wheel torque.
    - energy: penalty on squared joint torques.
    - idle_bonus: tiny bonus when no command is given (encourages staying still if asked)."""
    # progress reward (only when asking to move forward)
    cmd_speed = state.get('commanded_speed', 0.0)
    act_speed = state.get('ground_velocity', 0.0)
    progress = cmd_speed * act_speed if cmd_speed > 0.0 else 0.0
    progress_reward = 0.8 * progress

    # stability: large negative if leaning too far
    pitch = state.get('pitch', 0.0)
    pitch_thresh = 0.35  # ~20 deg
    if abs(pitch) > pitch_thresh:
        stability_pitch = -5.0
    else:
        stability_pick = 0.0
        stability_pitch = 0.0  # no penalty within threshold
    # correct variable name
    stability_pitch = -5.0 if abs(pitch) > pitch_thresh else 0.0

    # yaw alignment via cosine similarity (bounded [-1,1])
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    norm_cmd = math.hypot(cmd_yaw, 1e-8)
    norm_act = math.hypot(act_yaw, 1e-8)
    yaw_cos = (cmd_yaw * act_yaw) / (norm_cmd * norm_act) if norm_cmd * norm_act != 0 else 1.0
    yaw_align = 0.3 * yaw_cos

    # jerk proxy: penalize rapid torque changes (use wheel torques as proxy)
    jerk = 0.0
    for side in ('left', 'right'):
        t = state.get(f'{side}_wheel_torque', 0.0)
        jerk += t * t
    jerk_penalty = -0.0004 * jerk

    # energy cost (torque squared)
    energy = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            energy += t * t
    energy_cost = -0.00015 * energy

    # idle bonus: when no movement requested, reward staying put
    idle = 0.0
    if abs(cmd_speed) < 1e-3 and abs(cmd_yaw) < 1e-3:
        idle = 0.02 * (1.0 - min(abs(act_speed), 0.5) / 0.5)  # higher bonus if actually slow
    else:
        idle = 0.0

    return {
        'progress_reward': progress_reward,
        'stability_pitch': stability_pitch,
        'yaw_align': yaw_align,
        'jerk_penalty': jerk_penalty,
        'energy_cost': energy_cost,
        'idle_bonus': idle,
    }
