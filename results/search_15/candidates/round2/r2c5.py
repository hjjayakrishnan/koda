import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sparse‑shaped reward:
    - big_pos: +1.0 if pitch within ±0.1 rad AND ground speed within 0.05 m/s of command.
    - mediocre: +0.1 if pitch within ±0.2 rad.
    - penalty: -0.01 * (absolute pitch) to discourage large leans.
    - control_cost: -0.0005 * (sum of squared actions).
    """
    pitch = state.get('pitch', 0.0)
    sp_cmd = state.get('commanded_speed', 0.0)
    sp_act = state.get('ground_velocity', 0.0)
    speed_err = abs(sp_cmd - sp_act)

    # big reward when both conditions satisfied
    if abs(pitch) <= 0.1 and speed_err <= 0.05:
        big_pos = 1.0
    else:
        big_pos = 0.0

    # medium reward for moderate uprightness
    if abs(pitch) <= 0.2:
        mediocre = 0.1
    else:
        mediocre = 0.0

    # continuous lean penalty
    lean_pen = -0.01 * abs(pitch)

    # control effort cost
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    control_cost = -0.0005 * (ground_cmd * ground_cmd + yaw_cmd * yaw_cmd)

    return {
        'big_pos': big_pos,
        'mediocre': mediocre,
        'lean_penalty': lean_pen,
        'control_cost': control_cost,
    }
