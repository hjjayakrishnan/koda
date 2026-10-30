import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Progress‑and‑heading reward with smoothness regularisation.
    - forward_progress: ground_velocity * dt (meters moved).
    - heading_align: cosine of yaw to keep robot pointing forward.
    - pitch_penalty: squared pitch (keeps robot near vertical).
    - rate_smooth: penalises rapid changes in base rates (proxy for jerk).
    - action_reg: small penalty on magnitude of commanded velocities.
    """
    dt = state.get('dt', 0.01)
    ground_vel = state.get('ground_velocity', 0.0)
    forward_progress = ground_vel * dt

    yaw = state.get('yaw', 0.0)
    heading_align = math.cos(yaw)          # 1 when yaw≈0

    pitch = state.get('pitch', 0.0)
    pitch_penalty = -0.1 * pitch * pitch   # discourage lean

    # smoothness: penalise high pitch_rate and yaw_rate
    pr = state.get('pitch_rate', 0.0)
    yr = state.get('yaw_rate', 0.0)
    rate_smooth = -0.05 * (pr * pr + yr * yr)

    # action regularisation (encourage modest commands)
    gv_cmd = action.get('ground_velocity', 0.0)
    yv_cmd = action.get('yaw_velocity', 0.0)
    action_reg = -0.001 * (gv_cmd * gv_cmd + yv_cmd * yv_cmd)

    return {
        'forward_progress': forward_progress,
        'heading_align': heading_align,
        'pitch_penalty': pitch_penalty,
        'rate_smooth': rate_smooth,
        'action_reg': action_reg,
    }
