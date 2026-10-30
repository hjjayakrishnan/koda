import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sparse‑style reward:
    - arrival_bonus: large positive when both speed and yaw are within tight bands.
    - stability_penalty: negative when pitch exceeds a threshold.
    - jerk_penalty: approximates joint jerk via change in velocity (proxy = velocity^2).
    - control_effort: penalizes squared action magnitudes.
    - footing_reward: small reward per timestep with wheel contact.
    """
    # arrival bonus (only when both errors are tiny)
    sp_err = state.get('commanded_speed', 0.0) - state.get('ground_velocity', 0.0)
    y_err  = state.get('commanded_yaw_rate', 0.0) - state.get('yaw_rate', 0.0)
    inside = (abs(sp_err) < 0.05) and (abs(y_err) < 0.02)
    arrival_bonus = 5.0 if inside else 0.0

    # stability penalty (large negative if tipped)
    pitch = state.get('pitch', 0.0)
    stability_penalty = -2.0 if abs(pitch) > 0.3 else 0.0   # ~17 deg

    # jerk proxy: sum of squared joint velocities (higher vel → higher jerk)
    jerk = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            jerk += v * v
    jerk_penalty = -0.00005 * jerk

    # control effort: penalize magnitude of actions
    act_sp = action.get('ground_velocity', 0.0)
    act_yw = action.get('yaw_velocity', 0.0)
    control_effort = -0.001 * (act_sp * act_sp + act_yw * act_yw)

    # footing reward
    contact = state.get('contact', 0.0)
    footing_reward = 0.02 * contact

    return {
        'arrival_bonus': arrival_bonus,
        'stability_penalty': stability_penalty,
        'jerk_penalty': jerk_penalty,
        'control_effort': control_effort,
        'footing_reward': footing_reward,
    }
