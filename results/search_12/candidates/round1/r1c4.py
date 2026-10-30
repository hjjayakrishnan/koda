import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sigmoid‑shaped rewards:
    - balance_sig: rises as pitch approaches zero.
    - speed_sig: rises as ground velocity approaches commanded speed.
    - upright_bonus: extra reward when the robot is upright (|pitch| < 0.05) and moving forward.
    - torque_penalty: linear penalty on absolute torque sum.
    """
    pitch = state.get('pitch', 0.0)
    # Sigmoid centred at 0, steepness 20
    balance_sig = 1.0 / (1.0 + math.exp(20 * abs(pitch)))  # yields ~1 when pitch~0

    ground_vel = state.get('ground_velocity', 0.0)
    cmd_speed = state.get('commanded_speed', 0.0)
    speed_diff = ground_vel - cmd_speed
    speed_sig = 1.0 / (1.0 + math.exp(15 * abs(speed_diff)))  # ~1 when matched

    # Upright bonus: only if within tight band and forward velocity positive relative to command
    upright = 1.0 if abs(pitch) < 0.05 else 0.0
    forward_ok = 1.0 if (ground_vel * cmd_speed) >= 0 else 0.0  # same sign or both zero
    upright_bonus = 0.5 * upright * forward_ok

    # Torque penalty (linear to avoid exploding gradients)
    torque_abs_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            torque_abs_sum += abs(state.get(f'{side}_{joint}_torque', 0.0))
    torque_penalty = -0.01 * torque_abs_sum

    return {
        'balance_sig': balance_sig,
        'speed_sig': speed_sig,
        'upright_bonus': upright_bonus,
        'torque_penalty': torque_penalty,
    }
