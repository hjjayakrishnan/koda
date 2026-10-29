import math

def compute_reward(state: dict, action: dict) -> dict:
    """Gaussian‑shaped rewards for pitch and speed, plus a smooth‑action penalty."""
    r = {}
    # Pitch: reward highest when upright, falls off with sigma = 0.2 rad (~11°)
    pitch_err = state['pitch']
    r['pitch_gauss'] = math.exp(-(pitch_err * pitch_err) / (2.0 * 0.2 * 0.2)) - 1.0

    # Speed: similar Gaussian around commanded speed, sigma = 0.3 m/s
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_gauss'] = math.exp(-(speed_err * speed_err) / (2.0 * 0.3 * 0.3)) - 1.0

    # Yaw rate damping: penalize large yaw rates (helps heading stability)
    r['yaw_rate'] = -0.2 * math.tanh(abs(state['yaw_rate']))

    # Action smoothness: penalize large commanded velocities (encourage modest inputs)
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd    = action.get('yaw_velocity', 0.0)
    r['action'] = -0.01 * (ground_cmd * ground_cmd + yaw_cmd * yaw_cmd)

    # Contact bonus
    r['contact'] = 0.8 if state['contact'] > 0.5 else -0.5

    return r
