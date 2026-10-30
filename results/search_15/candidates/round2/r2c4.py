import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Components:
    - speed_ratio: reward proportional to actual/commanded speed (bounded).
    - yaw_alignment_l1: L1 penalty on yaw rate error.
    - stability_gauss: product of Gaussians for pitch and pitch_rate.
    - jerk_penalty: approximates discrete jerk via change in action.
    - heartbeat: tiny periodic signal to encourage policy variation.
    """
    # speed ratio (avoid division by zero)
    cmd = state.get('commanded_speed', 0.0)
    act = state.get('ground_velocity', 0.0)
    if cmd != 0.0:
        ratio = act / cmd
        # clip to reasonable range and reward being close to 1
        ratio = max(min(ratio, 2.0), -2.0)
        speed_ratio = -0.2 * (ratio - 1.0) * (ratio - 1.0)  # parabola around 1
    else:
        # when no command, reward staying still
        speed_ratio = -0.1 * act * act

    # yaw alignment L1
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    yaw_err = act_yaw - cmd_yaw
    yaw_alignment_l1 = -0.05 * abs(yaw_err)

    # stability as product of Gaussians (more restrictive)
    pitch = state.get('pitch', 0.0)
    pitch_rate = state.get('pitch_rate', 0.0)
    gauss_pitch = math.exp(-(pitch * pitch) / (2 * 0.15 * 0.15))
    gauss_rate = math.exp(-(pitch_rate * pitch_rate) / (2 * 0.5 * 0.5))
    stability_gauss = (gauss_pitch * gauss_rate) - 0.5  # shift so zero at rest gives ~0

    # jerk penalty: approximate derivative of action using previous step stored in state?
    # Since we lack memory, use magnitude of action as proxy for jerk (large actions cause jerk)
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    jerk_penalty = -0.0002 * (ground_cmd * ground_cmd + yaw_cmd * yaw_cmd)

    # heartbeat: small sinusoid based on time to break symmetry
    t = state.get('dt', 0.0) * 0.0  # placeholder; we don't have time, skip
    # Instead use a function of state that varies: sum of joint positions
    joint_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            joint_sum += state.get(f'{side}_{joint}_position', 0.0)
    heartbeat = 0.0001 * math.sin(joint_sum)

    return {
        'speed_ratio': speed_ratio,
        'yaw_alignment_l1': yaw_alignment_l1,
        'stability_gauss': stability_gauss,
        'jerk_penalty': jerk_penalty,
        'heartbeat': heartbeat,
    }
