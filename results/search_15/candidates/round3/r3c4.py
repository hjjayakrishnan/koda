import math

def compute_reward(state: dict, action: dict) -> dict:
    """
    Reward components:
    - speed_band: Reward if speed error lies within a tolerance band, else linear penalty.
    - yaw_band: Same for yaw rate.
    - upright_tanh: Smooth tanh-shaped reward for small pitch.
    - wheel_sync: Reward for left and right wheel velocities being similar (reduces slip).
    - joint_smooth: Penalty on joint acceleration approximated by change in velocity via dt? Not available → use joint velocity squared.
    - control_effort: Penalty on squared action values.
    """
    # Speed band reward
    cmd_speed = state.get('commanded_speed', 0.0)
    actual_speed = state.get('ground_velocity', 0.0)
    speed_err = cmd_speed - actual_speed
    band = 0.05  # m/s tolerance
    if abs(speed_err) <= band:
        speed_band = 0.1  # small reward for being inside band
    else:
        speed_band = -0.05 * (abs(speed_err) - band)  # linear penalty outside

    # Yaw band reward
    cmd_yaw = state.get('commanded_yaw_rate', 0.0)
    actual_yaw = state.get('yaw_rate', 0.0)
    yaw_err = cmd_yaw - actual_yaw
    yaw_band = 0.02
    if abs(yaw_err) <= yaw_band:
        yaw_band_reward = 0.05
    else:
        yaw_band_reward = -0.02 * (abs(yaw_err) - yaw_band)

    # Uprightness via tanh (smooth, saturates)
    pitch = state.get('pitch', 0.0)
    upright_tanh = math.tanh(1.0 / (abs(pitch) + 0.1))  # larger when pitch small
    # scale to roughly [0,1]
    upright_tanh = upright_tanh / math.tanh(1.0 / 0.1)

    # Wheel synchronization: reward when left and right wheel velocities match
    left_wheel_v = state.get('left_wheel_velocity', 0.0)
    right_wheel_v = state.get('right_wheel_velocity', 0.0)
    wheel_diff = left_wheel_v - right_wheel_v
    wheel_sync = -0.001 * (wheel_diff * wheel_diff)  # penalty for mismatch

    # Joint smoothness: penalize high joint velocities
    joint_vel_sq = 0.0
    for side in ('left', 'right'):
        for joint in ('hip', 'knee', 'wheel'):
            v = state.get(f'{side}_{joint}_velocity', 0.0)
            joint_vel_sq += v * v
    joint_smooth = -0.00008 * joint_vel_sq

    # Control effort: penalize squared action magnitudes
    ground_cmd = action.get('ground_velocity', 0.0)
    yaw_cmd = action.get('yaw_velocity', 0.0)
    control_effort = -0.0005 * (ground_cmd * ground_cmd + yaw_cmd * yaw_cmd)

    return {
        'speed_band': speed_band,
        'yaw_band': yaw_band_reward,
        'upright_tanh': upright_tanh,
        'wheel_sync': wheel_sync,
        'joint_smooth': joint_smooth,
        'control_effort': control_effort,
    }
