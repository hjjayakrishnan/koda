import math

def compute_reward(state: dict, action: dict) -> dict:
    """Novelty-driven reward:
    - forward_gain: linear reward for forward velocity up to a soft cap.
    - yaw_penalty: quadratic penalty for unwanted yaw when command is zero.
    - pitch_shield: soft‑constraint barrier that grows quickly near fall angle.
    - joint_limit_avoid: penalizes approaching joint limits (assume ±2.5 rad).
    - actuation_smooth: penalizes high joint accelerations approximated by joint torque.
    - contact_consistency: rewards continuous contact (both wheels) and punishes intermittent loss."""
    # forward gain with saturation
    vx = state.get('ground_velocity', 0.0)
    forward_gain = math.tanh(vx / 0.5)  # saturates around ±0.5 m/s

    # yaw penalty only when no yaw command
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    yaw_penalty = -0.5 * act_yaw * act_yaw if abs(cmd_yaw) < 1e-3 else 0.0

    # pitch shield: barrier function
    pitch = state.get('pitch', 0.0)
    max_pitch = 0.4  # ~23 degrees before fall
    if abs(pitch) >= max_pitch:
        pitch_shield = -10.0
    else:
        # grows as pitch approaches limit
        pitch_shield = -2.0 * (abs(pitch) / max_pitch) ** 2

    # joint limit avoidance (assuming safe range ±2.5 rad)
    limit = 2.5
    joint_avoid = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            pos = state.get(f'{side}_{joint}_position', 0.0)
            # distance to nearest limit
            dist = limit - abs(pos)
            if dist > 0:
                joint_avoid += -1.0 / (dist * dist + 1e-3)  # blows up near limit
            else:
                joint_avoid += -5.0  # already past limit (should not happen)
    joint_avoid *= 0.0005

    # actuation smoothness via torque (proxy for acceleration)
    smooth = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            smooth += t * t
    actuation_smooth = -0.0002 * smooth

    # contact consistency: reward when both wheels touch, punish flicker
    contact = state.get('contact', 0.0)  # 0,1 from env
    # assume contact==1 means at least one wheel; we approximate both by checking wheel velocities similarity
    lw = state.get('left_wheel_velocity', 0.0)
    rw = state.get('right_wheel_velocity', 0.0)
    both_contact_est = 1.0 if abs(lw - rw) < 0.1 else 0.0  # rough heuristic
    contact_cons = 0.03 * both_contact_est

    return {
        'forward_gain': forward_gain,
        'yaw_penalty': yaw_penalty,
        'pitch_shield': pitch_shield,
        'joint_avoid': joint_avoid,
        'actuation_smooth': actuation_smooth,
        'contact_consistency': contact_cons,
    }
