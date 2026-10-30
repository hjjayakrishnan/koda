import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - speed_l1:   negative L1 error on forward speed.
    - yaw_l1:     negative L1 error on yaw rate.
    - upright_cos: cosine‑based reward for staying near vertical (max 1 at pitch=0).
    - power_cost: estimated electrical power ≈ Σ |torque * velocity| for all actuators.
    - contact_ok: reward for having at least one wheel in contact.
    - action_smooth: penalty for large commanded velocities (discourages aggressive inputs).
    """
    # speed tracking (L1)
    cmd_v = state.get('commanded_speed', 0.0)
    act_v = state.get('ground_velocity', 0.0)
    speed_l1 = -abs(cmd_v - act_v)

    # yaw tracking (L1)
    cmd_ywr = state.get('commanded_yaw_rate', 0.0)
    act_ywr = state.get('yaw_rate', 0.0)
    yaw_l1 = -abs(cmd_ywr - act_ywr)

    # upright via cosine (bounded [0,1])
    pitch = state.get('pitch', 0.0)
    upright_cos = max(math.cos(pitch), 0.0)   # 1 when pitch=0, drops to 0 at ±π/2

    # power consumption approximation
    power = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            w = state.get(f'{side}_{joint}_velocity', 0.0)
            power += abs(t * w)
    power_cost = -0.0005 * power   # scale tuned to keep term modest

    # contact encouragement
    contact = state.get('contact', 0.0)
    contact_ok = 0.02 * contact   # 0.02 when both wheels touch, else 0

    # penalize large action magnitudes (helps learning stability)
    act_ground = action.get('ground_velocity', 0.0)
    act_yaw = action.get('yaw_velocity', 0.0)
    action_smooth = -0.001 * (act_ground * act_ground + act_yaw * act_yaw)

    return {
        'speed_l1': speed_l1,
        'yaw_l1': yaw_l1,
        'upright_cos': upright_cos,
        'power_cost': power_cost,
        'contact_ok': contact_ok,
        'action_smooth': action_smooth,
    }
