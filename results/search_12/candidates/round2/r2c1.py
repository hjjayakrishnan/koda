import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Simple tracking‑plus‑stability reward.
    - speed_error: negative squared difference between commanded and actual forward speed.
    - upright: cosine of pitch (1 when upright, 0 when sideways, negative when inverted).
    - torque_penalty: sum of squared joint torques scaled down.
    - contact_bonus: small reward for maintaining wheel‑ground contact.
    """
    # speed tracking
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    speed_error = -0.5 * speed_err * speed_err   # negative quadratic

    # uprightness (penalise lean)
    pitch = state.get('pitch', 0.0)
    upright = math.cos(pitch)  # 1 at pitch=0, decreases as |pitch| grows

    # torque effort
    torque_sum = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            t = state.get(f'{side}_{joint}_torque', 0.0)
            torque_sum += t * t
    torque_penalty = -0.0002 * torque_sum

    # contact encouragement
    contact = state.get('contact', 0.0)
    contact_bonus = 0.01 * contact   # 0.01 when both wheels touch, 0 otherwise

    return {
        'speed_error': speed_error,
        'upright': upright,
        'torque_penalty': torque_penalty,
        'contact_bonus': contact_bonus,
    }
