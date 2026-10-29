import math

def compute_reward(state: dict, action: dict) -> dict:
    """Soft‑penalty rewards for posture, speed and effort, with a small alive bonus."""
    r = {}
    # Posture: gentle tanh‑shaped penalty (≈linear for small angles, saturates)
    pitch_pen = math.tanh(abs(state['pitch']))
    yaw_pen   = math.tanh(abs(state['yaw']))   # discourage unnecessary turning
    r['posture'] = -0.8 * (pitch_pen + yaw_pen)

    # Speed tracking: L2 penalty softened by a small epsilon to avoid huge spikes
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed'] = -0.5 * (speed_err * speed_err) / (1.0 + abs(speed_err))

    # Effort: quadratic torque cost
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['effort'] = -0.0004 * torque_sq

    # Alive/contact bonus
    r['alive'] = 1.0 if state['contact'] > 0.5 else -1.0

    return r
