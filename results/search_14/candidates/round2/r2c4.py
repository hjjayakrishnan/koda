import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward focusing on smooth forward motion:
    - forward speed reward proportional to actual velocity (clipped).
    - penalty for deviating from commanded speed (linear).
    - yaw alignment reward: cosine similarity between desired yaw rate and actual.
    - pitch penalty (quadratic).
    - joint torque smoothness (difference between left/right torques).
    - contact bonus.
    """
    # forward speed reward (encourage moving forward)
    fwd = state.get('ground_velocity', 0.0)
    # clip negative values (reward only forward motion)
    fwd_reward = 0.08 * max(fwd, 0.0)

    # speed tracking linear penalty
    cmd_speed = state.get('commanded_speed', 0.0)
    speed_err = cmd_speed - fwd
    speed_lin_pen = -0.1 * abs(speed_err)

    # yaw alignment via cosine (approx using dot product normalized)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    act_yaw = state.get('yaw_rate', 0.0)
    # avoid division by zero
    norm = math.sqrt(max(cmd_yaw*cmd_yaw + 1e-8, 1e-8)) * math.sqrt(max(act_yaw*act_yaw + 1e-8, 1e-8))
    yaw_cos = (cmd_yaw * act_yaw) / norm if norm != 0 else 0.0
    yaw_align = 0.05 * yaw_cos  # max 0.05 when aligned

    # pitch penalty
    pitch = state.get('pitch', 0.0)
    pitch_pen = -0.04 * pitch * pitch

    # torque symmetry penalty (encourage equal left/right effort)
    left_torque = (state.get('left_hip_torque',0.0) +
                   state.get('left_knee_torque',0.0) +
                   state.get('left_wheel_torque',0.0))
    right_torque = (state.get('right_hip_torque',0.0) +
                    state.get('right_knee_torque',0.0) +
                    state.get('right_wheel_torque',0.0))
    torque_sym = -0.0002 * (left_torque - right_torque) ** 2

    # contact bonus
    contact = state.get('contact', 0.0)
    contact_bonus = 0.015 * contact

    return {
        'fwd_reward': fwd_reward,
        'speed_lin_pen': speed_lin_pen,
        'yaw_align': yaw_align,
        'pitch_pen': pitch_pen,
        'torque_sym': torque_sym,
        'contact_bonus': contact_bonus,
    }
