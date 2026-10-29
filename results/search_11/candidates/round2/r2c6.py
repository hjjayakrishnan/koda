import math

def compute_reward(state: dict, action: dict) -> dict:
    """Minimalist reward: survive + move forward, with soft limits on actuators."""
    r = {}
    # Survival: contact gives reward, falling loses big penalty
    r['survive'] = 2.0 if state['contact'] > 0.5 else -3.0

    # Forward velocity reward proportional to action (encourages using wheels)
    # Clip to avoid extreme values
    ground_cmd = action.get('ground_velocity', 0.0)
    # reward scales with commanded speed but saturates at 1.0 m/s command
    r['forward'] = math.tanh(abs(ground_cmd)) * 0.5  # max 0.5

    # Penalize large joint torques to prevent saturation
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['torque_penalty'] = -0.0004 * torque_sq

    # Small penalty for excessive pitch rate (helps damp oscillations)
    r['pitch_rate_pen'] = -0.2 * math.tanh(abs(state['pitch_rate']))

    # Yaw rate penalty to keep heading steady
    r['yaw_rate_pen'] = -0.1 * math.tanh(abs(state['yaw_rate']))

    return r
