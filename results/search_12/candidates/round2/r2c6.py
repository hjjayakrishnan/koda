import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Sparse‑style reward: give a bonus only when the robot is both upright enough
    and moving within a tolerance of the commanded speed; otherwise a small
    negative cost for actuation.
    - upright_reward: +0.04 if |pitch| < 0.2 rad (~11°) and contact == 1.
    - speed_reward: +0.03 if |cmd - vel| < 0.07 m/s.
    - actuation_cost: -0.0002 * sum of squared joint torques.
    - fallback_penalty: -0.001 applied every step (encourages achieving bonuses).
    """
    pitch = state.get('pitch', 0.0)
    contact = state.get('contact', 0.0)
    upright_ok = (abs(pitch) < 0.2) and (contact > 0.5)
    upright_reward = 0.04 if upright_ok else 0.0

    cmd = state.get('commanded_speed', 0.0)
    vel = state.get('ground_velocity', 0.0)
    speed_ok = abs(cmd - vel) < 0.07
    speed_reward = 0.03 if speed_ok else 0.0

    torque_sq = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sq += t * t
    actuation_cost = -0.0002 * torque_sq

    fallback_penalty = -0.001

    return {
        'upright_reward': upright_reward,
        'speed_reward': speed_reward,
        'actuation_cost': actuation_cost,
        'fallback_penalty': fallback_penalty,
    }
