import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Energy‑shaping reward:
    - total_energy: negative of approximated mechanical energy (kinetic + potential).
    - tracking: quadratic error on speed and yaw rate.
    - alignment: reward for zero lateral slip (approximated by yaw rate when no turn).
    - regularization: small L2 on joint velocities to keep motion smooth.
    """
    # Approximate kinetic energy: 0.5 * m * v^2 + 0.5 * I * omega^2
    # Use unit mass and inertia for scaling; only relative values matter.
    v = state.get('ground_velocity', 0.0)
    omega_yaw = state.get('yaw_rate', 0.0)
    ke = 0.5 * (v * v + 0.01 * omega_yaw * omega_yaw)  # small weight on yaw

    # Potential energy from pitch (height change approx proportional to sin(pitch))
    pitch = state.get('pitch', 0.0)
    pe = 0.5 * (pitch * pitch)  # simple quadratic approximation

    total_energy = -(ke + pe)  # we want low energy => negative reward

    # Tracking terms
    sp_err = state.get('commanded_speed', 0.0) - v
    yaw_err = state.get('commanded_yaw_rate', 0.0) - omega_yaw
    track = -0.4 * sp_err * sp_err - 0.2 * yaw_err * yaw_err

    # Alignment: when no turn command, reward low yaw rate
    if abs(state.get('commanded_yaw_rate', 0.0)) < 0.02:
        align = 0.02 * math.exp(- (omega_yaw * omega_yaw) / (2 * 0.1 * 0.1))
    else:
        align = 0.0

    # Joint velocity regularization (smooth motion)
    jv2 = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            jv2 += state.get(f'{side}_{joint}_velocity', 0.0) ** 2
    reg = -0.00005 * jv2

    return {
        'total_energy': total_energy,
        'tracking': track,
        'alignment': align,
        'regularization': reg,
    }
