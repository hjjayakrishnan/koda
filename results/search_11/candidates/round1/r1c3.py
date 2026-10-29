import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward based on exponential decay of errors and energy efficiency."""
    r = {}

    # Exponential penalty for pitch deviation (more forgiving near zero)
    pitch_err = abs(state['pitch'])
    r['pitch_exp'] = -1.0 * (math.exp(pitch_err) - 1.0)  # zero at err=0, grows fast

    # Exponential penalty for speed tracking error
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_exp'] = -0.8 * (math.exp(abs(speed_err)) - 1.0)

    # Energy cost approximated by sum of squared torques
    torque_sq = (state['left_hip_torque'] ** 2 + state['right_hip_torque'] ** 2 +
                 state['left_knee_torque'] ** 2 + state['right_knee_torque'] ** 2 +
                 state['left_wheel_torque'] ** 2 + state['right_wheel_torque'] ** 2)
    r['effort'] = -0.0005 * torque_sq

    # Contact encouragement (binary)
    r['contact'] = 1.0 if state['contact'] > 0.5 else -0.5

    return r
