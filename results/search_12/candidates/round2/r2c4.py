import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Energy‑shaped reward: maximize forward work while minimizing metabolic cost.
    - mechanical_power: forward force * velocity approximated as wheel_torque * wheel_velocity.
    - metabolic_cost: sum of squared torques (effort proxy).
    - orientation_cost: penalises pitch and yaw rates squared.
    - progress_bonus: small reward for moving forward when power > 0.
    """
    # mechanical power approximation (only wheel contributions)
    power = 0.0
    for side in ('left', 'right'):
        t_w = state.get(f'{side}_wheel_torque', 0.0)
        v_w = state.get(f'{side}_wheel_velocity', 0.0)
        power += t_w * v_w   # Nm * rad/s ≈ Watts

    # metabolic cost (effort)
    cost = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            tj = state.get(f'{side}_{joint}_torque', 0.0)
            cost += tj * tj
    metabolic = -0.0004 * cost

    # orientation cost (kinetic energy proxy)
    pr = state.get('pitch_rate', 0.0)
    yr = state.get('yaw_rate', 0.0)
    orient = -0.08 * (pr * pr + yr * yr)

    # progress bonus only when producing positive forward power
    progress_bonus = 0.005 * max(power, 0.0)

    return {
        'mechanical_power': power,
        'metabolic_cost': metabolic,
        'orientation_cost': orient,
        'progress_bonus': progress_bonus,
    }
