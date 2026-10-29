import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward that encourages natural gait and low effort while balancing."""
    r = {}

    # Upright reward using cosine – maximal when pitch≈0
    r['upright_cosine'] = 2.0 * math.cos(state['pitch'])  # cos(0)=1 → reward ~2

    # Penalize large joint velocities to avoid jerky motion
    vel_sq = (state['left_hip_velocity'] ** 2 + state['right_hip_velocity'] ** 2 +
              state['left_knee_velocity'] ** 2 + state['right_knee_velocity'] ** 2 +
              state['left_wheel_velocity'] ** 2 + state['right_wheel_velocity'] ** 2)
    r['velocity_penalty'] = -0.02 * vel_sq

    # Encourage forward progress toward commanded speed (linear reward)
    speed_diff = state['commanded_speed'] - state['ground_velocity']
    # Positive when slower than command → reward for catching up
    r['speed_progress'] = 1.5 * max(0.0, -speed_diff)  # reward only if lagging

    # Slight bonus for maintaining wheel contact
    r['contact_bonus'] = 0.5 if state['contact'] > 0.5 else 0.0

    return r
