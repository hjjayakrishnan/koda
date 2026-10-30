import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Energy‑aware forward progress:
    - progress: forward displacement multiplied by a cosines similarity with heading.
    - stability: inverse of kinetic energy in the base (penalizes high pitch/yaw rates).
    - efficiency: reward inversely proportional to total squared torque (like effort).
    - alive_bonus: tiny constant for being alive.
    """
    # Forward progress approximated by ground_velocity * dt
    dt = state.get('dt', 0.01)
    ground_vel = state.get('ground_velocity', 0.0)
    progress = ground_vel * dt  # meters moved this step

    # Heading alignment: if yaw deviates from zero, reduce forward progress reward
    yaw = state.get('yaw', 0.0)
    alignment = math.cos(yaw)  # 1 when yaw=0, reduces as yaw grows
    progress_aligned = progress * alignment

    # Stability: penalize base kinetic energy (0.5*I*omega^2) – use proxy with rates
    pitch_rate = state.get('pitch_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    ke_proxy = 0.5 * (pitch_rate * pitch_rate + yaw_rate * yaw_rate)
    stability = -0.1 * ke_proxy  # negative penalty

    # Efficiency: torque cost
    torque_sq_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            torque_sq_sum += state.get(f'{side}_{joint}_torque', 0.0) ** 2
    efficiency = -0.0005 * torque_sq_sum

    alive_bonus = 0.02

    return {
        'progress': progress_aligned,
        'stability': stability,
        'efficiency': efficiency,
        'alive_bonus': alive_bonus,
    }
