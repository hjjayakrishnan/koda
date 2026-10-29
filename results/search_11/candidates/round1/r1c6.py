import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward combining stability, forward motion, and directional alignment."""
    r = {}

    # Stability term: penalize pitch and its rate (like a PD controller)
    r['stability'] = -3.0 * state['pitch'] ** 2 - 0.8 * state['pitch_rate'] ** 2

    # Forward progress: reward moving in the direction of commanded speed
    # Use dot product of velocity and commanded direction (here 1‑D)
    progress = state['ground_velocity'] * state['commanded_speed']
    # Normalize by max possible speed to keep bounded; assume max 2 m/s
    max_speed = 2.0
    r['forward_progress'] = progress / max_speed  # positive when aligned

    # Yaw alignment: keep yaw near zero (robot starts facing forward)
    r['yaw_alignment'] = -1.0 * state['yaw'] ** 2

    # Control smoothness: penalize high wheel accelerations approximated by wheel velocity change
    # Use wheel velocities as a proxy; penalize large wheel speeds to avoid slipping
    wheel_speed_sq = state['left_wheel_velocity'] ** 2 + state['right_wheel_velocity'] ** 2
    r['wheel_speed_penalty'] = -0.005 * wheel_speed_sq

    # Contact maintenance (binary reward)
    r['contact'] = 0.5 if state['contact'] > 0.5 else -0.5

    return r
