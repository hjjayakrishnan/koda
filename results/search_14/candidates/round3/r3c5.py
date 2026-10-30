import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Terms:
    - speed_tanh: tanh-shaped reward for speed tracking (bounded).
    - yaw_tanh: same for yaw.
    - upright_cosine2: cosine squared reward for pitch (sharp peak).
    - jerk_penalty: approximate jerk via change in acceleration approximated by pitch_rate*dt.
    - action_rate_penalty: penalize large action values (disagree with previous step not stored, so use magnitude).
    - energy_const: small negative constant to discourage idle spinning.
    """
    dt = state.get('dt', 0.01)

    # speed tanh
    cmd_s = state.get('commanded_speed', 0.0)
    act_s = state.get('ground_velocity', 0.0)
    sp_err = cmd_s - act_s
    speed_tanh = math.tanh(-sp_err * sp_err * 0.5)  # negative, bounded [-1,0]

    # yaw tanh
    cmd_y = state.get('commanded_yaw_rate', 0.0)
    act_y = state.get('yaw_rate', 0.0)
    y_err = cmd_y - act_y
    yaw_tanh = math.tanh(-y_err * y_err * 0.5)

    # upright cosine squared (max 1 at pitch=0)
    pitch = state.get('pitch', 0.0)
    upright_cosine2 = math.cos(pitch) * math.cos(pitch)  # cos^2

    # jerk penalty: proxy using pitch_rate * dt (change in pitch per step)
    pitch_rate = state.get('pitch_rate', 0.0)
    jerk_penalty = -0.1 * (pitch_rate * dt) ** 2

    # action rate penalty (magnitude)
    gv = action.get('ground_velocity', 0.0)
    yv = action.get('yaw_velocity', 0.0)
    action_rate_penalty = -0.0005 * (gv * gv + yv * yv)

    # energy constant (encourage not wasting energy when no command)
    energy_const = -0.0002

    return {
        'speed_tanh': speed_tanh,
        'yaw_tanh': yaw_tanh,
        'upright_cosine2': upright_cosine2,
        'jerk_penalty': jerk_penalty,
        'action_rate_penalty': action_rate_penalty,
        'energy_const': energy_const,
    }
