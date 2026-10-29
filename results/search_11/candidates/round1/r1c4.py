import math

def compute_reward(state: dict, action: dict) -> dict:
    """Reward that combines pose stability, forward velocity, and yaw tracking."""
    r = {}

    # Pose stability: penalize both pitch and yaw deviations
    r['pose_penalty'] = -2.0 * (state['pitch'] ** 2 + state['yaw'] ** 2)

    # Forward velocity tracking (quadratic)
    speed_err = state['ground_velocity'] - state['commanded_speed']
    r['speed_tracking'] = -1.5 * speed_err ** 2

    # Yaw rate tracking (encourage following commanded yaw rate)
    yaw_rate_err = state['yaw_rate'] - state['commanded_yaw_rate']
    r['yaw_rate_tracking'] = -0.5 * yaw_rate_err ** 2

    # Damping term for joint accelerations (approximate via velocity change proxy)
    # Use current joint velocities as a proxy for smoothness
    joint_vel_sq = (state['left_hip_velocity'] ** 2 + state['right_hip_velocity'] ** 2 +
                    state['left_knee_velocity'] ** 2 + state['right_knee_velocity'] ** 2 +
                    state['left_wheel_velocity'] ** 2 + state['right_wheel_velocity'] ** 2)
    r['smoothness'] = -0.01 * joint_vel_sq

    # Contact bonus (keep wheels on ground)
    r['contact'] = 0.8 if state['contact'] > 0.5 else 0.0

    return r
