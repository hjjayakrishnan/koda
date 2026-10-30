import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - progress_toward_goal: reward proportional to forward speed when commanded speed > 0.
    - stand_still_bonus: small bonus when commanded speed is zero and robot stays near zero speed.
    - upright_exp: Gaussian reward for pitch (max at zero).
    - yaw_align: Reward for matching yaw direction to commanded yaw rate using cosine similarity.
    - energy_cost: Penalty on sum of squared torques and joint velocities.
    - contact_maintain: Bonus for continuous contact.
    """
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    # When we want to move forward, reward actual forward speed; clip negative to avoid rewarding backward motion
    if cmd_speed > 0.0:
        progress = min(actual_speed, cmd_speed)  # reward up to commanded speed
        progress_toward = 0.6 * progress
    else:
        # Want to stay still: reward low speed
        progress_toward = 0.2 * (-abs(actual_speed))  # negative penalty for any movement

    # Stand‑still bonus when cmd_speed≈0
    stand_still = 0.0
    if abs(cmd_speed) < 0.02:
        stand_still = 0.05 * math.exp(-(actual_speed * actual_speed) / (2 * 0.02 * 0.02))

    # Upright Gaussian
    pitch = state.get('pitch', 0.0)
    upright_exp = math.exp(-(pitch * pitch) / (2 * 0.25 * 0.25))  # sigma=0.25 rad

    # Yaw alignment via cosine of angle difference (assuming small angles, cos ≈ 1 - 0.5*err^2)
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    yaw_rate = state.get('yaw_rate', 0.0)
    yaw_diff = cmd_yaw - yaw_rate
    yaw_align = math.cos(yaw_diff)  # ranges [-1,1]; shift to [0,1]
    yaw_align = max(0.0, yaw_align)

    # Energy cost: torques + joint velocities
    energy = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            energy += 0.0001 * t * t + 0.00005 * v * v
    energy_cost = -energy

    # Contact maintenance
    contact = state.get('contact', 0.0)
    contact_maintain = 0.009 * contact

    return {
        'progress_toward': progress_toward,
        'stand_still_bonus': stand_still,
        'upright_exp': upright_exp,
        'yaw_align': yaw_align,
        'energy_cost': energy_cost,
        'contact_maintain': contact_maintain,
    }
